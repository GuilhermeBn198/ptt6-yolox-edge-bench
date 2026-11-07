idf_component_register(SRCS "main.cpp" "model_data.cpp"
                       INCLUDE_DIRS "."
                       PRIV_REQUIRES esp_driver_gpio esp_timer driver
                       EMBED_FILES "test_image.bin")

