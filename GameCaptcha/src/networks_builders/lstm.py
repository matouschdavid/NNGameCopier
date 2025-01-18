from keras import layers, Model
def build_combined_lstm(latent_shape, input_dim, input_prominence, time_dim, sequence_length):
    # Latent shape from encoder
    height, width, channels = latent_shape

    # Inputs
    encoder_input = layers.Input(shape=(sequence_length, height, width, channels), name="encoder_input")
    input_vector = layers.Input(shape=(sequence_length, input_dim), name="input_vector")
    time_input = layers.Input(shape=(sequence_length, time_dim), name="time_input")

    # ConvLSTM for encoder output
    conv_lstm_output = layers.ConvLSTM2D(
        filters=32, kernel_size=(3, 3), padding="same", return_sequences=True, name="conv_lstm"
    )(encoder_input)

    # Flatten the ConvLSTM output while keeping the sequence dimension
    flat_conv_lstm_output = layers.TimeDistributed(layers.Flatten(), name="flatten_conv_lstm")(conv_lstm_output)

    # Repeat the flattened input vector across sequence length
    # repeated_input_vector = layers.RepeatVector(input_prominence, name="Repeat Input")(input_vector)

    # Combine ConvLSTM output with the repeated input vector
    combined_main_input = layers.Concatenate(name="combine_inputs")([flat_conv_lstm_output, input_vector, time_input])

    # LSTM for combined input
    x = layers.LSTM(128, return_sequences=True)(combined_main_input)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)

    # Final output layer
    latent_dim = height * width * channels  # Dimension of the latent representation from the encoder
    total_input_dim = latent_dim + input_dim  # Combining latent representation and input vector
    output_dim = total_input_dim + time_dim  # Final output dimension

    # Output layer for the LSTM model
    lstm_outputs = layers.Dense(output_dim, name="output_dense")(x)

    # Define and compile the model
    lstm_model = Model(inputs=[encoder_input, input_vector, time_input], outputs=lstm_outputs, name="combined_lstm_model")
    lstm_model.compile(optimizer="adam", loss="mse")

    return lstm_model


