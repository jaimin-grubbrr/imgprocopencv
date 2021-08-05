package com.example.imgprocopencv;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.opencv.imgproc.Imgproc.THRESH_BINARY;

public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2  {

    private CameraBridgeViewBase   mOpenCvCameraView;
    private static final String    TAG                 = "Detection::MainActivity";
    private Mat mRgba;
    private Mat mHsv;


    private boolean isSaved = false;
    private int frameCount = 0;

    private List<Mat> frames = new ArrayList<>();

    private Mat firstFrame = null;
    private Mat secondFrame = null;
    private Bitmap backImg = null;

    private long timeStamp = System.currentTimeMillis();
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);


        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.java_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);


        backImg = drawableToBitmap(getDrawable(R.drawable.img_back_1));
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(width, height, CvType.CV_8UC4);
        mHsv = new Mat(width, height, CvType.CV_8UC3);

        firstFrame=mRgba;

        Utils.bitmapToMat(backImg,firstFrame);

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {


        if (System.currentTimeMillis() > timeStamp){
            process(inputFrame.rgba());
            timeStamp=timeStamp+2000;
            Log.e("Called","process");
        }

        return inputFrame.rgba();
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }
    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }



    private List<Mat> getChannel(Mat rgbMat){
        List<Mat> channels = new ArrayList<>();
        Core.split(rgbMat, channels);
        return channels;
    }

    private Mat applyGaussianBlur(Mat mat){
        Mat updateMat = new Mat(mat.rows(),mat.cols(),mat.type());
        Imgproc.GaussianBlur(mat,updateMat,new Size(15, 15),0);
        return updateMat;
    }
    private Bitmap getBitmap(Mat mat){
        Bitmap.Config conf = Bitmap.Config.ARGB_8888; // see other conf types
        Bitmap bitmap = Bitmap.createBitmap(mat.width(),mat.height(),conf);
        Utils.matToBitmap(mat,bitmap);
        return bitmap;
    }
    private Mat findAbsDiff(Mat mat1,Mat mat2){
        Mat desMat = new Mat(mat1.rows(),mat1.cols(),mat1.type());
        Core.absdiff(mat1,mat2,desMat);
        return desMat;
    }

    private Mat findThreshold(Mat mat){
        Mat desMat = new Mat(mat.rows(),mat.cols(),mat.type());
        Imgproc.threshold(mat,desMat,40,255,THRESH_BINARY);
        return desMat;
    }


    private void process(Mat secondFrame){

        if (firstFrame == null | secondFrame == null){
            Log.e("Frame","Not get");
            return;
        }


        Mat frame1 = firstFrame;
        Mat frame2 = secondFrame;

        Mat redFrame1 = getChannel(frame1).get(0);
        Mat blueFrame1 = getChannel(frame1).get(1);
        Mat greenFrame1 = getChannel(frame1).get(2);

        Mat redFrame2 = getChannel(frame2).get(0);
        Mat blueFrame2 = getChannel(frame2).get(1);
        Mat greenFrame2 = getChannel(frame2).get(2);

        Mat redBlurFrame1Gaussian = applyGaussianBlur(redFrame1);
        Mat blueBlurFrame1Gaussian = applyGaussianBlur(blueFrame1);
        Mat greenBlurFrame1Gaussian = applyGaussianBlur(greenFrame1);

        Mat redBlurFrame2Gaussian = applyGaussianBlur(redFrame2);
        Mat blueBlurFrame2Gaussian = applyGaussianBlur(blueFrame2);
        Mat greenBlurFrame2Gaussian = applyGaussianBlur(greenFrame2);

        Mat redDif = findAbsDiff(redBlurFrame1Gaussian,redBlurFrame2Gaussian);
        Mat blueDif = findAbsDiff(blueBlurFrame1Gaussian,blueBlurFrame2Gaussian);
        Mat greenDif = findAbsDiff(greenBlurFrame1Gaussian,greenBlurFrame2Gaussian);

        Mat thresholdRed  = findThreshold(redDif);

        getContourArea(thresholdRed);


     //   Imgproc.drawContours();
    }

    private static ArrayList<Rect> getContourArea(Mat mat) {
        Mat hierarchy = new Mat();
        Mat image = mat.clone();
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(image, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        Rect rect = null;
        double maxArea = 300;
        ArrayList<Rect> arr = new ArrayList<Rect>();
        for (int i = 0; i < contours.size(); i++) {
            Mat contour = contours.get(i);
            double contourArea = Imgproc.contourArea(contour);
            if (contourArea > maxArea) {
                rect = Imgproc.boundingRect(contours.get(i));
                Log.e("Rect","x-"+rect.x+", y-"+rect.y+" h-"+rect.height+", w-"+rect.width);
                arr.add(rect);
            }
        }
        return arr;
    }


    private void SaveImage(Mat red, Mat blue, Mat green) {

        Mat blueBlur = applyGaussianBlur(blue);
        Mat redBlur = applyGaussianBlur(red);
        Mat greenBlur = applyGaussianBlur(green);


        Bitmap.Config conf = Bitmap.Config.ARGB_8888; // see other conf types
        Bitmap bitmap = Bitmap.createBitmap(red.width(),red.height(),conf);
        Bitmap bitmapBlur = Bitmap.createBitmap(red.width(),red.height(),conf);

        Mat updateMat = new Mat(red.rows(),red.cols(),red.type());

        Imgproc.GaussianBlur(red,updateMat,new Size(15, 15),0);

        Utils.matToBitmap(red,bitmap);
        Utils.matToBitmap(updateMat,bitmapBlur);

        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        File myDir = new File(root + "/saved_images");
        if (!myDir.exists()){
            myDir.mkdirs();
        }

        String fname = "real.jpg";
        File file = new File (myDir, fname);
        if (file.exists ()) file.delete ();

        String fname1 = "blur.jpg";
        File blurFile = new File (myDir, fname1);
        if (blurFile.exists ()) blurFile.delete ();

        try {
            FileOutputStream outBlur = new FileOutputStream(blurFile);
            bitmapBlur.compress(Bitmap.CompressFormat.JPEG, 100, outBlur);
            outBlur.flush();
            outBlur.close();

            FileOutputStream out = new FileOutputStream(file);
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
            out.flush();
            out.close();

            isSaved=true;

        } catch (Exception e) {
            e.printStackTrace();
        }
    }


    public static Bitmap drawableToBitmap (Drawable drawable) {
        Bitmap bitmap = null;

        if (drawable instanceof BitmapDrawable) {
            BitmapDrawable bitmapDrawable = (BitmapDrawable) drawable;
            if(bitmapDrawable.getBitmap() != null) {
                return bitmapDrawable.getBitmap();
            }
        }

        if(drawable.getIntrinsicWidth() <= 0 || drawable.getIntrinsicHeight() <= 0) {
            bitmap = Bitmap.createBitmap(1, 1, Bitmap.Config.ARGB_8888); // Single color bitmap will be created of 1x1 pixel
        } else {
            bitmap = Bitmap.createBitmap(drawable.getIntrinsicWidth(), drawable.getIntrinsicHeight(), Bitmap.Config.ARGB_8888);
        }

        Canvas canvas = new Canvas(bitmap);
        drawable.setBounds(0, 0, canvas.getWidth(), canvas.getHeight());
        drawable.draw(canvas);
        return bitmap;
    }
}