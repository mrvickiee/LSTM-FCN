
TRAIN_FILES = ['../data\\Adiac_TRAIN', # 0
               '../data\\ArrowHead_TRAIN',  # 1
               '../data\\ChlorineConcentration_TRAIN',  # 2
               '../data\\InsectWingbeatSound_TRAIN',  # 3
               '../data\\Lighting7_TRAIN',  # 4
               '../data\\Wine_TRAIN',  # 5
               '../data\\WordsSynonyms_TRAIN',  # 6
               '../data\\50words_TRAIN',  # 7
               '../data\\Beef_TRAIN',  # 8
               '../data\\DistalPhalanxOutlineAgeGroup_TRAIN',  # 9 (inverted dataset)
               '../data\\DistalPhalanxOutlineCorrect_TRAIN',  # 10 (not inverted dataset)
               '../data\\DistalPhalanxTW_TRAIN',  # 11 (inverted dataset)
               '../data\\ECG200_TRAIN',  # 12
               '../data\\ECGFiveDays_TRAIN',  # 13
               '../data\\BeetleFly_TRAIN',  # 14
               '../data\\BirdChicken_TRAIN',  # 15
               '../data\\ItalyPowerDemand_TRAIN', # 16
               '../data\\SonyAIBORobotSurface_TRAIN', # 17
               '../data\\SonyAIBORobotSurfaceII_TRAIN', # 18
               '../data\\MiddlePhalanxOutlineAgeGroup_TRAIN', # 19
               '../data\\MiddlePhalanxOutlineCorrect_TRAIN', # 20
               '../data\\MiddlePhalanxTW_TRAIN', # 21
               '../data\\ProximalPhalanxOutlineAgeGroup_TRAIN', # 22
               '../data\\ProximalPhalanxOutlineCorrect_TRAIN', # 23
               '../data\\ProximalPhalanxTW_TRAIN', # 24 (inverted dataset)
               '../data\\MoteStrain_TRAIN', # 25
               '../data\\MedicalImages_TRAIN', # 26
               '../data\\Strawberry_TEST',  # 27 (inverted dataset)
               '../data\\ToeSegmentation1_TRAIN',  # 28
               '../data\\Coffee_TRAIN',  # 29
               '../data\\Cricket_X_TRAIN',  # 30
               '../data\\Cricket_Y_TRAIN',  # 31
               '../data\\Cricket_Z_TRAIN',  # 32
               '../data\\uWaveGestureLibrary_X_TRAIN',  # 33
               '../data\\uWaveGestureLibrary_Y_TRAIN',  # 34
               '../data\\uWaveGestureLibrary_Z_TRAIN',  # 35
               '../data\\ToeSegmentation2_TRAIN',  # 36
               '../data\\DiatomSizeReduction_TRAIN',  # 37
               '../data\\car_TRAIN',  # 38
               '../data\\CBF_TRAIN',  # 39
               '../data\\CinC_ECG_torso_TRAIN',  # 40
               '../data\\Computers_TRAIN',  # 41
               '../data\\Earthquakes_TRAIN',  # 42 (not inverted dataset)
               '../data\\ECG5000_TRAIN',  # 43
               '../data\\ElectricDevices_TRAIN',  # 44
               '../data\\FaceAll_TRAIN',  # 45
               '../data\\FaceFour_TRAIN',  # 46
               '../data\\FacesUCR_TRAIN',  # 47
               '../data\\Fish_TRAIN',  # 48
               '../data\\FordA_TRAIN',  # 49 (not inverted dataset)
               '../data\\FordB_TRAIN',  # 50 (not inverted dataset)
               '../data\\Gun_Point_TRAIN',  # 51
               '../data\\Ham_TRAIN',  # 52
               '../data\\HandOutlines_TRAIN',  # 53 (not inverted dataset)
               '../data\\Haptics_TRAIN', # 54
               '../data\\Herring_TRAIN', # 55
               '../data\\InlineSkate_TRAIN', # 56
               '../data\\LargeKitchenAppliances_TRAIN', # 57
               '../data\\Lighting2_TRAIN', # 58
               '../data\\MALLAT_TRAIN', # 59
               '../data\\Meat_TRAIN', # 60
               '../data\\NonInvasiveFatalECG_Thorax1_TRAIN', # 61
               '../data\\NonInvasiveFatalECG_Thorax2_TRAIN', # 62
               '../data\\OliveOil_TRAIN', # 63
               '../data\\OSULeaf_TRAIN', # 64
               '../data\\PhalangesOutlinesCorrect_TRAIN',  # 65
               '../data\\Phoneme_TRAIN',  # 66
               '../data\\plane_TRAIN',  # 67
               '../data\\RefrigerationDevices_TRAIN',  # 68
               '../data\\ScreenType_TRAIN',  # 69
               '../data\\ShapeletSim_TRAIN',  # 70
               '../data\\ShapesAll_TRAIN',  # 71
               '../data\\SmallKitchenAppliances_TRAIN',  # 72
               '../data\\StarlightCurves_TRAIN',  # 73
               '../data\\SwedishLeaf_TRAIN',  # 74
               '../data\\Symbols_TRAIN',  # 75
               '../data\\synthetic_control_TRAIN',  # 76
               '../data\\Trace_TRAIN',  # 77
               '../data\\Two_Patterns_TRAIN',  # 78
               '../data\\TwoLeadECG_TRAIN',  # 79
               '../data\\UWaveGestureLibraryAll_TRAIN',  # 80
               '../data\\wafer_TRAIN',  # 81
               '../data\\Worms_TRAIN',  # 82
               '../data\\WormsTwoClass_TRAIN',  # 83
               '../data\\yoga_TRAIN',  # 84,
               '../master-by-phil/data/UCR-dataset/fmri/fmri_1_TRAIN', #85
               '../master-by-phil/data/UCR-dataset/fmri/fmri_1_TRAIN', #86
               '../master-by-phil/data/UCR-dataset/fmri/fmri_2_TRAIN', #87
               '../master-by-phil/data/UCR-dataset/fmri/fmri_2_TRAIN', #88
               '../master-by-phil/data/UCR-dataset/fmri/fmri_3_TRAIN', #89
               '../master-by-phil/data/UCR-dataset/fmri/fmri_3_TRAIN', #90
               '../master-by-phil/data/UCR-dataset/fmri/fmri_4_TRAIN', #91
               '../master-by-phil/data/UCR-dataset/fmri/fmri_4_TRAIN', #92
               '../master-by-phil/data/UCR-dataset/fmri/fmri_5_TRAIN', #93
               '../master-by-phil/data/UCR-dataset/fmri/fmri_5_TRAIN', #94
               ]

TEST_FILES = ['../data\\Adiac_TEST', # 0
              '../data\\ArrowHead_TEST', # 1
              '../data\\ChlorineConcentration_TEST', # 2
              '../data\\InsectWingbeatSound_TEST', # 3
              '../data\\Lighting7_TEST', # 4
              '../data\\Wine_TEST', # 5
              '../data\\WordsSynonyms_TEST', # 6
              '../data\\50words_TEST', # 7
              '../data\\Beef_TEST', # 8
              '../data\\DistalPhalanxOutlineAgeGroup_TEST', # 9 (not inverted dataset)
              '../data\\DistalPhalanxOutlineCorrect_TEST', # 10 (not inverted dataset)
              '../data\\DistalPhalanxTW_TEST', # 11 (not inverted dataset)
              '../data\\ECG200_TEST', # 12
              '../data\\ECGFiveDays_TEST', # 13
              '../data\\BeetleFly_TEST', # 14
              '../data\\BirdChicken_TEST', # 15
              '../data\\ItalyPowerDemand_TEST', # 16
              '../data\\SonyAIBORobotSurface_TEST', # 17
              '../data\\SonyAIBORobotSurfaceII_TEST', # 18
              '../data\\MiddlePhalanxOutlineAgeGroup_TEST', # 19 (inverted dataset)
              '../data\\MiddlePhalanxOutlineCorrect_TEST', # 20 (inverted dataset)
              '../data\\MiddlePhalanxTW_TEST', # 21 (inverted dataset)
              '../data\\ProximalPhalanxOutlineAgeGroup_TEST', # 22
              '../data\\ProximalPhalanxOutlineCorrect_TEST', # 23
              '../data\\ProximalPhalanxTW_TEST', # 24 (inverted dataset)
              '../data\\MoteStrain_TEST', # 25
              '../data\\MedicalImages_TEST', # 26
              '../data\\Strawberry_TRAIN',  # 27
              '../data\\ToeSegmentation1_TEST',  # 28
              '../data\\Coffee_TEST',  # 29
              '../data\\Cricket_X_TEST',  # 30
              '../data\\Cricket_Y_TEST',  # 31
              '../data\\Cricket_Z_TEST',  # 32
              '../data\\uWaveGestureLibrary_X_TEST',  # 33
              '../data\\uWaveGestureLibrary_Y_TEST',  # 34
              '../data\\uWaveGestureLibrary_Z_TEST',  # 35
              '../data\\ToeSegmentation2_TEST',  # 36
              '../data\\DiatomSizeReduction_TEST',  # 37
              '../data\\car_TEST',  # 38
              '../data\\CBF_TEST',  # 39
              '../data\\CinC_ECG_torso_TEST',  # 40
              '../data\\Computers_TEST',  # 41
              '../data\\Earthquakes_TEST',  # 42 (not inverted dataset)
              '../data\\ECG5000_TEST',  # 43
              '../data\\ElectricDevices_TEST',  # 44
              '../data\\FaceAll_TEST',  # 45
              '../data\\FaceFour_TEST',  # 46
              '../data\\FacesUCR_TEST',  # 47
              '../data\\Fish_TEST',  # 48
              '../data\\FordA_TEST',  # 49 (not inverted dataset)
              '../data\\FordB_TEST',  # 50 (not inverted dataset)
              '../data\\Gun_Point_TEST',  # 51
              '../data\\Ham_TEST',  # 52
              '../data\\HandOutlines_TEST',  # 53 (not inverted dataset)
              '../data\\Haptics_TEST',  # 54
              '../data\\Herring_TEST',  # 55
              '../data\\InlineSkate_TEST',  # 56
              '../data\\LargeKitchenAppliances_TEST',  # 57
              '../data\\Lighting2_TEST',  # 58
              '../data\\MALLAT_TEST',  # 59
              '../data\\Meat_TEST',  # 60
              '../data\\NonInvasiveFatalECG_Thorax1_TEST',  # 61
              '../data\\NonInvasiveFatalECG_Thorax2_TEST',  # 62
              '../data\\OliveOil_TEST',  # 63
              '../data\\OSULeaf_TEST',  # 64
              '../data\\PhalangesOutlinesCorrect_TEST',  # 65
              '../data\\Phoneme_TEST',  # 66
              '../data\\plane_TEST',  # 67
              '../data\\RefrigerationDevices_TEST',  # 68
              '../data\\ScreenType_TEST',  # 69
              '../data\\ShapeletSim_TEST',  # 70
              '../data\\ShapesAll_TEST',  # 71
              '../data\\SmallKitchenAppliances_TEST',  # 72
              '../data\\StarlightCurves_TEST',  # 73
              '../data\\SwedishLeaf_TEST',  # 74
              '../data\\Symbols_TEST',  # 75
              '../data\\synthetic_control_TEST',  # 76
              '../data\\Trace_TEST',  # 77
              '../data\\Two_Patterns_TEST',  # 78
              '../data\\TwoLeadECG_TEST',  # 79
              '../data\\UWaveGestureLibraryAll_TEST',  # 80
              '../data\\wafer_TEST',  # 81
              '../data\\Worms_TEST',  # 82
              '../data\\WormsTwoClass_TEST',  # 83
              '../data\\yoga_TEST',  # 84,
              '../master-by-phil/data/UCR-dataset/fmri/fmri_1_VAL', #85
              '../master-by-phil/data/UCR-dataset/fmri/fmri_1_TEST', #86
              '../master-by-phil/data/UCR-dataset/fmri/fmri_2_VAL', #87
              '../master-by-phil/data/UCR-dataset/fmri/fmri_2_TEST', #88
              '../master-by-phil/data/UCR-dataset/fmri/fmri_3_VAL', #89
              '../master-by-phil/data/UCR-dataset/fmri/fmri_3_TEST', #90
              '../master-by-phil/data/UCR-dataset/fmri/fmri_4_VAL', #91
              '../master-by-phil/data/UCR-dataset/fmri/fmri_4_TEST', #92
              '../master-by-phil/data/UCR-dataset/fmri/fmri_5_VAL', #93
              '../master-by-phil/data/UCR-dataset/fmri/fmri_5_TEST', #94
              ]

MAX_SEQUENCE_LENGTH_LIST = [176, # 0
                            251, # 1
                            166, # 2
                            256, # 3
                            319, # 4
                            234, # 5
                            270, # 6
                            270, # 7
                            470, # 8
                            80,  # 9
                            80,  # 10
                            80,  # 11
                            96, # 12
                            136, # 13
                            512, # 14
                            512, # 15
                            24, # 16
                            70, # 17
                            65, # 18
                            80, # 19
                            80, # 20
                            80, # 21
                            80, # 22
                            80, # 23
                            80, # 24
                            84, # 25
                            99, # 26
                            235, # 27
                            277, # 28
                            286, # 29
                            300, # 30
                            300, # 31
                            300, # 32
                            315, # 33
                            315, # 34
                            315, # 35
                            343, # 36
                            345, # 37
                            577, # 38
                            128, # 39
                            1639, # 40
                            720, # 41
                            512, # 42
                            140, # 43
                            96, # 44
                            131, # 45
                            350, # 46
                            131, # 47
                            463, # 48
                            500, # 49
                            500, # 50
                            150, # 51
                            431, # 52
                            2709, # 53
                            1092, # 54
                            512, # 55
                            1882, # 56
                            720, # 57
                            637, # 58
                            1024, # 59
                            448, # 60
                            750, # 61
                            750, # 62
                            570, # 63
                            427, # 64
                            80, # 65
                            1024, # 66
                            144, # 67
                            720, # 68
                            720, # 69
                            500, # 70
                            512, # 71
                            720, # 72
                            1024, # 73
                            128, # 74
                            398, # 75
                            60, # 76
                            275, # 77
                            128, # 78
                            82, # 79
                            945, # 80
                            152, # 81
                            900, # 82
                            900, # 83
                            426, # 84
                            46, # 85
                            46, # 86
                            46, # 87
                            46, # 88
                            46, # 89
                            46, # 90
                            46, # 91
                            46, # 92
                            46, # 93
                            46, # 94
                            ]

NB_CLASSES_LIST = [37, # 0
                   3, # 1
                   3, # 2
                   11, # 3
                   7, # 4
                   2, # 5
                   25, # 6
                   50, # 7
                   5, # 8
                   3, # 9
                   2, # 10
                   6, # 11
                   2, # 12
                   2, # 13
                   2, # 14
                   2, # 15
                   2, # 16
                   2, # 17
                   2, # 18
                   3, # 19
                   2, # 20
                   6, # 21
                   3, # 22
                   2, # 23
                   6, # 24
                   2, # 25
                   10, # 26
                   2, # 27
                   2, # 28
                   2, # 29
                   12, # 30
                   12, # 31
                   12, # 32
                   8, # 33
                   8, # 34
                   8, # 35
                   2, # 36
                   4, # 37
                   4, # 38
                   3, # 39
                   4, # 40
                   2, # 41
                   2, # 42
                   5, # 43
                   7, # 44
                   14, # 45
                   4, # 46
                   14, # 47
                   7, # 48
                   2, # 49
                   2, # 50
                   2, # 51
                   2, # 52
                   2, # 53
                   5, # 54
                   2, # 55
                   7, # 56
                   3, # 57
                   2, # 58
                   8, # 59
                   3, # 60
                   42, # 61
                   42, # 62
                   4, # 63
                   6, # 64
                   2, # 65
                   39, # 66
                   7, # 67
                   3, # 68
                   3, # 69
                   2, # 70
                   60, # 71
                   3, # 72
                   3, # 73
                   15, # 74
                   6, # 75
                   6, # 76
                   4, # 77
                   4, # 78
                   2, # 79
                   8, # 80
                   2, # 81
                   5, # 82
                   2, # 83
                   2, # 84
                   5, # 85
                   5, # 86
                   5, # 87
                   5, # 88
                   5, # 89
                   5, # 90
                   5, # 91
                   5, # 92
                   5, # 93
                   5, # 94
                   ]
