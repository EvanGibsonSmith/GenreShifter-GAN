
I couldn't get the berkeley link to work from the GitHub, but I used kaggle. 
This can be done by searching up the dataset like the horse to zebra dataset and downloading manually
or by using a downloading script.

## Issue Log

Dataset would not load from old link, so I used Kaggle instead (with a script in my case) and put the
data in the proper format.

Some of the downloaded images were not in RGB mode. For this reason I altered the ImageDataset class to 
convert to RGB so the grayscale images could be processed. Otherwise, an issue of a mismatch in the number
of the channels for the model occured.

Needed to add name = main at the top of the train and test scripts for multithreading on Windows.