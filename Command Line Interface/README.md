To get MARS metric & MARS charts using command line interface. Execute below commands.

1. python mars_shinethrough.py shinethrough your_classifier_results.csv
Takes as input a .csv file with classifier results, formatted as describe in section 2.1. Outputs a text file with MARS ShineThrough scores and a .png file with the MARS ShineThrough chart.


2. python mars_occlusion.py occlusion your_classifier_results.csv 
Takes as input a .csv file with classifier results, formatted as describe in section 2.1. Outputs a text file with MARS Occlusion scores and a .png file with the MARS Occlusion chart.
