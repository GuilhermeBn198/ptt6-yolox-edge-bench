#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  Configurador Completo do Cartão SD - Raspberry Pi Zero 2 W"
echo "============================================================"

if [ "$(id -u)" -ne 0 ]; then
    echo "[ERRO] Execute este script com sudo:"
    echo "  sudo WIFI_SSID='SuaRede' WIFI_PSK='SuaSenha' ./raspberry/pi_zero_2w/configure_sdcard.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PUBKEY_FILE="${PUBKEY_FILE:-$SCRIPT_DIR/id_ed25519.pub}"
WIFI_SSID="${WIFI_SSID:-REDE_WIFI}"
WIFI_PSK="${WIFI_PSK:-SENHA_WIFI}"
RPI_USER="${RPI_USER:-filipe}"
RPI_USER_PASS="${RPI_USER_PASS:-raspberry}"

PUBKEY_CONTENT=""
if [ -f "$PUBKEY_FILE" ]; then
    PUBKEY_CONTENT="$(cat "$PUBKEY_FILE")"
fi

if [ ! -b /dev/sda1 ] || [ ! -b /dev/sda2 ]; then
    echo "[1/6] Aguardando o Raspberry Pi em modo rpiboot..."
    echo "      (Se necessário: desconecte o USB, puxe levemente o cartão SD,"
    echo "       reconecte o cabo na porta USB e empurre o cartão SD de volta)"
    rpiboot -v
    echo "      Aguardando partições /dev/sda1 e /dev/sda2 aparecerem..."
    for i in $(seq 1 30); do
        if [ -b /dev/sda1 ] && [ -b /dev/sda2 ]; then
            break
        fi
        sleep 1
    done
fi

sleep 2

BOOT_MNT="$(lsblk -no MOUNTPOINT /dev/sda1 | head -n1)"
ROOT_MNT="$(lsblk -no MOUNTPOINT /dev/sda2 | head -n1)"

if [ -z "$BOOT_MNT" ]; then
    BOOT_MNT="/mnt/rpi_bootfs"
    mkdir -p "$BOOT_MNT"
    mount /dev/sda1 "$BOOT_MNT"
fi

if [ -z "$ROOT_MNT" ]; then
    ROOT_MNT="/mnt/rpi_rootfs"
    mkdir -p "$ROOT_MNT"
    mount /dev/sda2 "$ROOT_MNT"
fi

echo "[2/6] Partições montadas em:"
echo "      BOOT: $BOOT_MNT"
echo "      ROOT: $ROOT_MNT"

# 1. Limpar cache do cloud-init para forçar execução limpa
echo "[3/6] Limpando cache antigo do cloud-init e desbloqueando rfkill..."
rm -rf "$ROOT_MNT/var/lib/cloud/"*
mkdir -p "$ROOT_MNT/var/lib/systemd/rfkill"
for f in "$ROOT_MNT/var/lib/systemd/rfkill/"*; do
    [ -f "$f" ] && echo 0 > "$f"
done

# 2. Atualizar meta-data com ambos instance-id e instance_id
cat > "$BOOT_MNT/meta-data" <<'EOF'
dsmode: local
instance-id: rpios-ptt6-v3
instance_id: rpios-ptt6-v3
local-hostname: raspberrypi
EOF

# 3. Garantir config.txt com dwc2 peripheral
if ! grep -q "dtoverlay=dwc2,dr_mode=peripheral" "$BOOT_MNT/config.txt"; then
    sed -i 's/^dtoverlay=dwc2$/dtoverlay=dwc2,dr_mode=peripheral/' "$BOOT_MNT/config.txt" || true
    if ! grep -q "dtoverlay=dwc2,dr_mode=peripheral" "$BOOT_MNT/config.txt"; then
        printf "\n[all]\ndtoverlay=dwc2,dr_mode=peripheral\n" >> "$BOOT_MNT/config.txt"
    fi
fi

# 4. Garantir módulos e regdom no cmdline.txt
echo "g_ether" > "$ROOT_MNT/etc/modules-load.d/usb-gadget.conf"

# 5. Criar perfil Wi-Fi e USB diretamente no NetworkManager do rootfs
echo "[4/6] Configurando Wi-Fi e USB Gadget diretamente no NetworkManager..."
NM_DIR="$ROOT_MNT/etc/NetworkManager/system-connections"
mkdir -p "$NM_DIR"
rm -f "$NM_DIR/"*

cat > "$NM_DIR/WiFi-PTT6.nmconnection" <<EOF
[connection]
id=$WIFI_SSID
uuid=7c4c90c9-215d-4a51-9988-0123456789ab
type=wifi
interface-name=wlan0
autoconnect=true
autoconnect-priority=100

[wifi]
mode=infrastructure
ssid=$WIFI_SSID

[wifi-security]
key-mgmt=wpa-psk
psk=$WIFI_PSK

[ipv4]
method=auto

[ipv6]
method=auto
EOF

cat > "$NM_DIR/USB Gadget (shared).nmconnection" <<'EOF'
[connection]
id=USB Gadget (shared)
uuid=8d5d90c9-215d-4a51-9988-0123456789ac
type=ethernet
interface-name=usb0
autoconnect=true
autoconnect-priority=50

[ipv4]
method=shared
address1=10.12.194.1/28

[ipv6]
method=disabled
EOF

chmod 600 "$NM_DIR/"*.nmconnection
chown -R 0:0 "$NM_DIR"

# 6. Criar serviço systemd nativo que garante usuário, SSH, USB0 e Wi-Fi em todo boot
echo "[5/6] Instalando serviço systemd ptt6-boot-setup e copiando arquivos do benchmark..."
cat > "$ROOT_MNT/usr/local/sbin/ptt6-boot-setup.sh" <<EOF
#!/bin/sh
# Desbloquear Wi-Fi
for f in /var/lib/systemd/rfkill/*; do
    [ -f "\$f" ] && echo 0 > "\$f"
done
rfkill unblock all || true
iw reg set BR || true

# Garantir usuário com UID 1000
if ! id $RPI_USER >/dev/null 2>&1; then
    useradd -m -s /bin/bash -G sudo,gpio,adm,dialout,plugdev,netdev $RPI_USER || true
fi
echo "$RPI_USER:$RPI_USER_PASS" | chpasswd || true
usermod -s /bin/bash -U $RPI_USER || true
echo "$RPI_USER ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/010_ptt6-nopasswd
chmod 0440 /etc/sudoers.d/010_ptt6-nopasswd

# Configurar chave SSH
mkdir -p /home/$RPI_USER/.ssh
echo "$PUBKEY_CONTENT" > /home/$RPI_USER/.ssh/authorized_keys
chmod 700 /home/$RPI_USER/.ssh
chmod 600 /home/$RPI_USER/.ssh/authorized_keys
chown -R $RPI_USER:$RPI_USER /home/$RPI_USER

# Garantir SSH ativo
systemctl enable ssh >/dev/null 2>&1 || true
systemctl restart ssh >/dev/null 2>&1 || true

# Ativar USB Gadget (usb0 -> 10.12.194.1/28)
modprobe dwc2 || true
modprobe g_ether || true
ip link set usb0 up || true
/usr/bin/rpi-usb-gadget on -f || true
ip addr add 10.12.194.1/28 dev usb0 2>/dev/null || true
ip link set usb0 up || true

# Ativar Wi-Fi
nmcli radio wifi on || true
nmcli connection reload || true
nmcli connection up "USB Gadget (shared)" || true
nmcli connection up "$WIFI_SSID" || true
EOF
chmod 755 "$ROOT_MNT/usr/local/sbin/ptt6-boot-setup.sh"

cat > "$ROOT_MNT/etc/systemd/system/ptt6-boot-setup.service" <<'EOF'
[Unit]
Description=PTT6 Boot Setup (User, SSH, USB Gadget, Wi-Fi)
After=network-pre.target NetworkManager.service systemd-rfkill.service
Wants=NetworkManager.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/ptt6-boot-setup.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

mkdir -p "$ROOT_MNT/etc/systemd/system/multi-user.target.wants"
ln -sf /etc/systemd/system/ptt6-boot-setup.service "$ROOT_MNT/etc/systemd/system/multi-user.target.wants/ptt6-boot-setup.service"
if [ -f "$ROOT_MNT/lib/systemd/system/ssh.service" ]; then
    ln -sf /lib/systemd/system/ssh.service "$ROOT_MNT/etc/systemd/system/multi-user.target.wants/ssh.service"
elif [ -f "$ROOT_MNT/usr/lib/systemd/system/ssh.service" ]; then
    ln -sf /usr/lib/systemd/system/ssh.service "$ROOT_MNT/etc/systemd/system/multi-user.target.wants/ssh.service"
fi

# Copiar arquivos de benchmark para /home/pi/ppe_benchmark
USER_HOME="$ROOT_MNT/home/pi"
[ -d "$ROOT_MNT/home/$RPI_USER" ] && USER_HOME="$ROOT_MNT/home/$RPI_USER"
mkdir -p "$USER_HOME/ppe_benchmark/models" "$USER_HOME/ppe_benchmark/raspberry/pi_zero_2w" "$USER_HOME/ppe_benchmark/data/raw"
cp "$SCRIPT_DIR/benchmark_yolox.py" "$SCRIPT_DIR/run_all_v13_experiments.py" "$SCRIPT_DIR/setup_rpi.sh" "$USER_HOME/ppe_benchmark/raspberry/pi_zero_2w/"
[ -f "$SCRIPT_DIR/test_image.jpg" ] && cp "$SCRIPT_DIR/test_image.jpg" "$USER_HOME/ppe_benchmark/raspberry/pi_zero_2w/"
[ -f "$REPO_DIR/models/yolox_nano_ppe_int8.tflite" ] && cp "$REPO_DIR/models/yolox_nano_ppe_int8.tflite" "$USER_HOME/ppe_benchmark/models/"
[ -f "$REPO_DIR/models/yolox_nano_ppe_float32.tflite" ] && cp "$REPO_DIR/models/yolox_nano_ppe_float32.tflite" "$USER_HOME/ppe_benchmark/models/"
chmod +x "$USER_HOME/ppe_benchmark/raspberry/pi_zero_2w/setup_rpi.sh"
chown -R 1000:1000 "$USER_HOME"

echo "[6/6] Sincronizando e desmontando partições..."
sync
umount /dev/sda1 2>/dev/null || true
umount /dev/sda2 2>/dev/null || true

echo "============================================================"
echo "  CONCLUÍDO COM SUCESSO!"
echo "  Agora desconecte o cabo USB, aguarde 3s e reconecte-o"
echo "  na porta do meio (USB) com o cartão MicroSD encaixado."
echo "============================================================"

