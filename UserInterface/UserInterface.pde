import processing.video.*;

Capture cam;
int CaptureCount = 0;

void setup() {
    size(1280, 960);

    String[] cameras = Capture.list();
    
    println("hello!");

    if (cameras.length == 0) {
     println("There are no cameras available for capture.");
     exit();
    } else {
       println("Available cameras:");
       for (int i = 0; i < cameras.length; i++) {
       println(cameras[i]);
       }
        
       // The camera can be initialized directly using an 
       // element from the array returned by list():
       cam = new Capture(this, cameras[0]);
       cam.start();     
    }
}

void draw() {
    if (cam.available() == true){
        cam.read();
    }
    pushMatrix();
    scale(-1, 1);
    image(cam, -cam.width * 2, 0, 1280, 960);
    popMatrix();

    if (mousePressed){
        println(mouseX, mouseY);
        String filename = str(CaptureCount) + "-" + str(mouseX) + "-" + str(mouseY) + ".png";
        saveFrame(filename);
        CaptureCount += 1;
    }
}
