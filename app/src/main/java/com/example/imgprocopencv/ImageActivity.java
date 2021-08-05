package com.example.imgprocopencv;

import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

public class ImageActivity extends AppCompatActivity implements View.OnClickListener {

    ImageView srcImageView,destImageView;
    private static final String    TAG                 = "ImageActivity";

    Button btnBlur,btnDiff,btnThreshold,btnContour;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image);

        srcImageView = findViewById(R.id.srcImage);
        destImageView = findViewById(R.id.destImage);

        btnBlur=findViewById(R.id.btnBlur);
        btnDiff=findViewById(R.id.btnDiff);
        btnThreshold=findViewById(R.id.btnThreshold);
        btnContour=findViewById(R.id.btnContour);

        btnBlur.setOnClickListener(this);

    }
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i("OpenCV", "OpenCV loaded successfully");
                //    imageMat=new Mat();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };



    @Override
    public void onPause()
    {
        super.onPause();

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

    public void DifferenceOfGaussian(Mat originalMat) {
        Mat grayMat = new Mat();
        Mat blur1 = new Mat();
        Mat blur2 = new Mat();

        //Converting the image to grayscale
        Imgproc.cvtColor(originalMat, grayMat, Imgproc.COLOR_BGR2GRAY);

        Imgproc.GaussianBlur(grayMat, blur1, new Size(15, 15), 5);
        Imgproc.GaussianBlur(grayMat, blur2, new Size(21, 21), 5);

        //Subtracting the two blurred images
        Mat DoG = new Mat();
        Core.absdiff(blur1, blur2, DoG);

        //Inverse Binary Thresholding
        Core.multiply(DoG, new Scalar(100), DoG);
        Imgproc.threshold(DoG, DoG, 50, 255, Imgproc.THRESH_BINARY_INV);

        Bitmap currentBitmap = Bitmap.createBitmap(originalMat.width(),originalMat.height(), Bitmap.Config.ARGB_8888);
        //Converting Mat back to Bitmap
        Utils.matToBitmap(DoG, currentBitmap);
        destImageView.setImageBitmap(currentBitmap);
    }

    @Override
    public void onClick(View v) {

        switch (v.getId()){
            case R.id.btnBlur:
                BitmapDrawable drawable = (BitmapDrawable) srcImageView.getDrawable();
                Bitmap bitmap = drawable.getBitmap();
                convertToBlackWhite(bitmap);
                break;
            case R.id.btnDiff:
                break;
            case R.id.btnThreshold:
                break;
            case R.id.btnContour:
                break;
            default:
                break;
        }
    }


    public void convertToBlackWhite(Bitmap compressImage)
    {
        if (compressImage == null){
            return;
        }

        Log.d("CV", "Before converting to black");
        Mat imageMat = new Mat();
        Utils.bitmapToMat(compressImage, imageMat);
        ///Imgproc.cvtColor(imageMat, imageMat, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(imageMat, imageMat, new Size(15, 15), 0);
        //Imgproc.adaptiveThreshold(imageMat, imageMat, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 5, 4);
        //Imgproc.medianBlur(imageMat, imageMat, 3);
        //Imgproc.threshold(imageMat, imageMat, 0, 255, Imgproc.THRESH_OTSU);

        Bitmap newBitmap = Bitmap.createBitmap(imageMat.width(),imageMat.height(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(imageMat, newBitmap);
        destImageView.setImageBitmap(newBitmap);
        Log.d("CV", "After converting to black");

    }
}
