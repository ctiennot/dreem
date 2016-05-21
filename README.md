# dreem
Challenge Dreem


/!\ Please make sure the folder data actually contains the following files: /!\
		- quality_dataset.h5
		- record1.h5
		- record2.h5
		- record1_bad_quality.csv
		- record1_good_quality.csv

/!\ Code is to be run with src folder as the working directory. /!\


Files descriptions :
--------------------

- easy_import.py: contains extract_signals function to load record1.h5 and record2.h5.

- exploratory.py: exploratory analysis of quality_dataset.h5.

- functions.py: a bunch of useful functions.

- quality_detector.py: implementation of the final model to predict signal quality on record1.h5 and record2.h5.

- randomForest.py: file to build a Random Forest classifier unsing quality_dataset.h5.

- incrementsVarRF.py: file to build a Random Forest classifier unsing quality_dataset.h5 after extraction piece-wise variances as features.

- RFacf.py: file to build a Random Forest classifier unsing quality_dataset.h5 after extraction piece-wise variances and ACF (Auto Correlation Functions) as features.