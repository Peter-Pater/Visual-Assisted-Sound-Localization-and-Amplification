import processing.video.*;
import processing.sound.*;
Movie myMovie;
SoundFile mySound;

//Capture cam;
int CaptureCount = 0;
boolean beamforming = false;
float paused_pos = 0;

void setup() {
    size(1280, 720);
    myMovie = new Movie(this, "video.mov");
    myMovie.play(); 
    mySound = new SoundFile(this, "audio_mono.wav");
    mySound.play();
}

void draw() {
    if (beamforming){
        fill(0, 0, 0, 80);
        rect(0, 0, width, height);
        textSize(50);
        fill(255, 255, 255, 200);
        text("Processing...", width / 2 - 100, height / 2);
        switchAudio();
    }else{
        image(myMovie, 0, 0);
    }
}

void movieEvent(Movie m){
    m.read();
}

// // for camera capture
// void setup() {
//     size(640, 480);

//     String[] cameras = Capture.list();
    
//     println("hello!");

//     if (cameras.length == 0) {
//      println("There are no cameras available for capture.");
//      exit();
//     } else {
//        println("Available cameras:");
//        for (int i = 0; i < cameras.length; i++) {
//        println(cameras[i]);
//        }
        
//        // The camera can be initialized directly using an 
//        // element from the array returned by list():
//        cam = new Capture(this, cameras[0]);
//        cam.start();     
//     }
// }

// // for camera capture
// void draw() {
//     if (cam.available() == true){
//         cam.read();
//     }
//     pushMatrix();
//     scale(-1, 1);
//     image(cam, -640, 0, 640, 480);
//     popMatrix();
// }

void mouseReleased(){
    println(mouseX, mouseY);
    String filename = str(CaptureCount) + "-" + str(mouseX) + "-" + str(mouseY) + ".png";
    saveFrame(filename);
    CaptureCount += 1;
    
    // pause video and audio
    myMovie.pause();
    paused_pos = mySound.position();
    mySound.stop();
    beamforming = true;
}

void switchAudio(){
      try{
          mySound = new SoundFile(this, "audio_processed.wav");
          mySound.cue(paused_pos);
          println("read success");
          mySound.play();
          myMovie.play();
          beamforming = false;
      }catch (Exception e) {
          println("file is still loading");
      }
}
