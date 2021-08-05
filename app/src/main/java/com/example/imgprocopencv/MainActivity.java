package com.example.imgprocopencv;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.os.Bundle;
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
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

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
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);


        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.java_surface_view);
        mOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
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
    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        List<Mat> channels = new ArrayList<>();
        Core.split(inputFrame.rgba(), channels);
        Mat red = channels.get(0);
        Mat blue = channels.get(1);
        Mat green = channels.get(2);

        List<Mat> c = new ArrayList<>();

        c.add(red);
        c.add(blue);
        c.add(green);

//        Core.absdiff();
//        Imgproc.threshold()
        Core.merge(c,mRgba);

        if (frameCount != 2){
            frames.add(mRgba);
            Log.e("Frame"," count "+frameCount);
            frameCount++;
        }

        if (frames.size() == 2){
            process();
        }

/*
        if (!isSaved){
            SaveImage(mRgba,blue,green);
        }
*/

        return mRgba;
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


    private void process(){

        Mat frame1 = frames.get(0);
        Mat frame2 = frames.get(1);

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

        Mat threshold  = findThreshold(redDif);



        threshold.get(1,1);
        //findContours();


    }

}