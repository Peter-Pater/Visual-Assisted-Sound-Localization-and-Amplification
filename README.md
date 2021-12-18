# Visual-Assisted-Sound-Localization-and-Amplification

## Simulated Experiments and Evaluations
In /micarray and evalution/evaluation you can find four notebooks, and each of them implements beamforming in a simulated environment and evaluated a test case (see the final report for more details)

The files are:
- Evaluation_Case1_R1R2.ipynb
- Evaluation_Case2_R1BG.ipynb
- Evaluation_Case3_FltBG.ipynb
- Evaluation_Case4_R1BGPosition.ipynb

These notebooks are huge, so we also prepared their correspondent html file in the same folder with all the results printed:
- Evaluation_Case1_R1R2.html
- Evaluation_Case2_R1BG.html
- Evaluation_Case3_FltBG.html
- Evaluation_Case4_R1BGPosition.html

## The Application in Real-world
The application in real-world does not work well, but you can run it as the following:
- go to /UserInterface
- download Processing 4
- run UserInterface.pde by double clicking on it
- run imageParse.py at the same time in a terminal

When a user clicks on the userinterface opened in processing, monodepth will translate the pixel coordinate to 3D position, and the python script will run beamforming and replace the audio. Due to the poor performance caused by the small microphone array we used, the difference is not audible.
