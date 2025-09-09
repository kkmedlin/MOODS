# MOODS
The goal of MOODS is to improve the classification of imbalanced data marred by the low representation of its minority class. To achieve its objective, MOODS resamples data. It leverages a multi-objective bilevel optimization framework to guide both synthetic oversampling and majority undersampling and find an optimal subsample of data able to, ultimately, mitigate model bias. So far, MOODS has been successfully deployed on 7 benchmark imbalanced datasets achieving F1 scores results up to 15% over SOTA.

Five code files:

    main - Before launching the main file, choose the dataset, title, number of runs, and number of steps.
    moods - Before running moods, choose either the data (for Abalone, Shuttle, Spambase, or Connect4) or smallData (for Ecoli, Yeast, or Winequality) file.
    data - For four larger datasets Gisette, Abolone, Spambase, Connect4. "Turn on" lines 306-308, 312-313, and 319-329 if using Gisette dataset. Comment out these lines if NOT using Gisette.
    smallData - For three smaller datasets Ecoli, Yeast, Winequality
    model - MOODS's convoluted neural network model; developed to match that of state-of-the-art methods SMOTified-GAN, GBO, SSG, and MUBO.
