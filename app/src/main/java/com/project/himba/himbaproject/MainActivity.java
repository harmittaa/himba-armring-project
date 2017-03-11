package com.project.himba.himbaproject;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.annotation.RequiresApi;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final int MY_PERMISSIONS_REQUEST_CAMERA = 645;
    private static final int MY_PERMISSION_REQUEST_WRITE = 644;
    private static final int MIN_MATCH_THRESHOLD = 0;
    private static final int MAX_MATCH_THRESHOLD = 100;
    private static final String TAG = "MainActivity";
    private Mat mRgba;
    private Mat mRgbaF;
    private Mat mRgbaT;
    private ImageView mSrcImg;
    private ImageView mRefImg;
    private File photoFile;
    private String photoPath;
    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private Bitmap currentPhoto;
    private Button takePhotoButton;
    private Button getDescriptorsButton;
    private Button takeComparisonImageButton;
    private boolean takingComparisonImage = false;
    private boolean comparisonImageChanged = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        mSrcImg = (ImageView) findViewById(R.id.userImage);
        mRefImg = (ImageView) findViewById(R.id.mRefImg);
        takePhotoButton = (Button) findViewById(R.id.takePhotoButton);
        takeComparisonImageButton = (Button) findViewById(R.id.takeComparisonImageButton);
        getDescriptorsButton = (Button) findViewById(R.id.getDescriptors);
        //setSupportActionBar(toolbar);

        if (!OpenCVLoader.initDebug()) {
            Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");

        } else {
            Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
        }

        checkHardwareCamera(this);

        requestExternalStorage(this);

       /* checkHardwareCamera(this);

        mOpenCvCameraView = (JavaCameraView) findViewById(R.id.openCvId);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);*/

        takePhotoButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                takingComparisonImage = false;
                dispatchTakePictureIntent();
            }
        });

        takeComparisonImageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                takingComparisonImage = true;
                dispatchTakePictureIntent();
            }
        });

        getDescriptorsButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                matchImages();
            }
        });

        //matchImages();
    }

    @RequiresApi(api = Build.VERSION_CODES.M)
    private void checkWritePermission() {
        if (checkSelfPermission(android.Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
            Log.e(TAG, "Permission is granted");
            //File write logic here
        } else {
            Log.e(TAG, "Permission is NOT granted");
            requestExternalStorage(this);
        }

    }

    private void matchImages() {
        Mat image1 = new Mat();
        Mat image2 = new Mat();
        MatOfKeyPoint refKeypoints = new MatOfKeyPoint();
        MatOfKeyPoint srcKeypoints = new MatOfKeyPoint();

        MatOfPoint2f reference = new MatOfPoint2f();
        MatOfPoint2f source = new MatOfPoint2f();
        Mat descriptors = new Mat();
        Mat logoDescriptors = new Mat();
        MatOfDMatch matches = new MatOfDMatch();
        MatOfDMatch goodMatches = new MatOfDMatch();
        LinkedList<DMatch> listOfGoodMatches = new LinkedList<>();
        LinkedList<Point> refObjectList = new LinkedList<>();
        LinkedList<Point> srcObjectList = new LinkedList<>();
        long time = System.currentTimeMillis();
        long time2 = System.currentTimeMillis();

        FeatureDetector featureDetector = FeatureDetector.create(FeatureDetector.BRISK);
        Bitmap bmImage1;

        if (comparisonImageChanged) {
            bmImage1 = ((BitmapDrawable) mRefImg.getDrawable()).getBitmap();
        } else {
            bmImage1 = BitmapFactory.decodeResource(this.getResources(), R.drawable.himba2);
        }
        // Fetch images from Drawables
        //Bitmap bmImage2 = BitmapFactory.decodeResource(this.getResources(), R.drawable.rings_hand);
        Bitmap bmImage2 = ((BitmapDrawable) mSrcImg.getDrawable()).getBitmap();
        // Change Bitmaps to Mats
        Utils.bitmapToMat(bmImage1, image1);
        Utils.bitmapToMat(bmImage2, image2);

        Imgproc.cvtColor(image1, image1, Imgproc.COLOR_RGBA2RGB);
        Imgproc.cvtColor(image2, image2, Imgproc.COLOR_RGBA2RGB);

        // Initialize the descriptorExtractor

        DescriptorExtractor descriptorExtractor = DescriptorExtractor.create(DescriptorExtractor.BRISK);

        //extract keypoints
        featureDetector.detect(image1, refKeypoints);
        Log.e(TAG, "number of BOATS Keypoints= " + refKeypoints.size());
        featureDetector.detect(image2, srcKeypoints);
        Log.e(TAG, "number of BEER Keypoints= " + srcKeypoints.size());
        Log.e(TAG, "keypoint calculation time elapsed" + (System.currentTimeMillis() - time));
        Log.e("LOG!", "logo type" + image2.type() + "  intype" + image1.type());

        //Extract descriptors
        descriptorExtractor.compute(image1, refKeypoints, descriptors);
        descriptorExtractor.compute(image2, srcKeypoints, logoDescriptors);
        Log.e("LOG!", "Description time elapsed" + (System.currentTimeMillis() - time2));

        // Create the matches
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        matcher.match(descriptors, logoDescriptors, matches);
        Log.e(TAG, "getDescriptorsButton: " + matches.toArray().length);

        double max_dist = 0;
        double min_dist = 35;
        List<DMatch> matchesList = matches.toList();

        for (int i = 0; i < descriptors.rows(); i++) {
            Double distance = (double) matchesList.get(i).distance;
            if (distance < min_dist) min_dist = distance;
            if (distance > max_dist) max_dist = distance;
        }

        for (int i = 0; i < descriptors.rows(); i++) {
            if (matchesList.get(i).distance < 3 * min_dist) {
                listOfGoodMatches.add(matchesList.get(i));
            }
        }

        goodMatches.fromList(listOfGoodMatches);

        List<KeyPoint> refObjectListKeypoints = refKeypoints.toList();
        List<KeyPoint> srcObjectListKeypoints = srcKeypoints.toList();

        Log.e(TAG, "getDescriptorsButton: EMPTY? = " + listOfGoodMatches.size());
        Log.e(TAG, "getDescriptorsButton: EMPTY 2 ? = " + refObjectListKeypoints.size());

        for (int i = 0; i < listOfGoodMatches.size(); i++) {
            refObjectList.addLast(refObjectListKeypoints.get(listOfGoodMatches.get(i).queryIdx).pt);
            srcObjectList.addLast(srcObjectListKeypoints.get(listOfGoodMatches.get(i).trainIdx).pt);
        }

        reference.fromList(refObjectList);
        source.fromList(srcObjectList);

        String result;

        Log.e(TAG, "getDescriptorsButton: Tresholds between these values: " + listOfGoodMatches.size());

        if (listOfGoodMatches.size() >= MIN_MATCH_THRESHOLD && listOfGoodMatches.size() < MAX_MATCH_THRESHOLD) {
            result = "They MATCH!";
        } else {
            result = "They DON'T match!";
        }

        Log.e(TAG, "getDescriptorsButton: RESULT" + result);

        Mat outputImage = new Mat();
        Bitmap comboBmp = combineImages(bmImage1, bmImage2);
        Log.e(TAG, "getDescriptorsButton: heigth of combo : " + comboBmp.getHeight());
        Utils.bitmapToMat(comboBmp, outputImage);

        Features2d.drawMatches(image1, refKeypoints, image2, srcKeypoints, goodMatches, outputImage);

        Bitmap bitmap = Bitmap.createBitmap(outputImage.cols(), outputImage.rows(), Bitmap.Config.ARGB_8888);
        /*mRefImg.setImageBitmap(comboBmp);
        mRefImg.invalidate();*/

        Utils.matToBitmap(outputImage, bitmap);

        /*mSrcImg.setImageBitmap(bitmap);
        mSrcImg.invalidate();*/
        Log.e(TAG, "getDescriptorsButton: heigth of bitmap : " + bitmap.getHeight());

        MediaStore.Images.Media.insertImage(getContentResolver(), comboBmp, "combo", "combo");
        MediaStore.Images.Media.insertImage(getContentResolver(), bitmap, "drawMatches", "drawmatches");
        Log.e(TAG, "getDescriptorsButton: Files are saved");

        new AlertDialog.Builder(this)
                .setTitle("Finished")
                .setMessage("Comaparison finished with " + listOfGoodMatches.size() + " matches")
                .setPositiveButton(android.R.string.yes, new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.cancel();
                    }
                }).setIcon(android.R.drawable.ic_dialog_alert).show();
    }

    public Bitmap combineImages(Bitmap c, Bitmap s) { // can add a 3rd parameter 'String loc' if you want to save the new image - left some code to do that at the bottom
        Bitmap cs = null;

        int width, height = 0;

        if (c.getWidth() > s.getWidth()) {
            width = c.getWidth() + s.getWidth();
            height = c.getHeight();
        } else {
            width = s.getWidth() + s.getWidth();
            height = c.getHeight();
        }

        cs = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

        Canvas comboImage = new Canvas(cs);

        comboImage.drawBitmap(c, 0f, 0f, null);
        comboImage.drawBitmap(s, c.getWidth(), 0f, null);

        // this is an extra bit I added, just incase you want to save the new image somewhere and then return the location
    /*String tmpImg = String.valueOf(System.currentTimeMillis()) + ".png";

    OutputStream os = null;
    try {
      os = new FileOutputStream(loc + tmpImg);
      cs.compress(CompressFormat.PNG, 100, os);
    } catch(IOException e) {
      Log.e("combineImages", "problem combining images", e);
    }*/

        return cs;
    }

    private boolean checkHardwareCamera(Context context) {
        if (context.getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA)) {
            Log.e("matti", "checkHardwareCamera: This device has a camera");
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                    MY_PERMISSION_REQUEST_WRITE);
            return true;
        } else {
            Log.e("matti", "checkHardwareCamera: This device doesn't have a camera");
            return false;
        }
    }

    private boolean requestExternalStorage(Context context) {
        int permissionCheck = ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                MY_PERMISSION_REQUEST_WRITE);

        return true;
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.e(TAG, "OpenCV loaded successfully");
                    //mOpenCvCameraView.enableView();
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    /**
     * This method is invoked when camera preview has started. After this method is invoked
     * the frames will start to be delivered to client via the onCameraFrame() callback.
     *
     * @param width  -  the width of the frames that will be delivered
     * @param height - the height of the frames that will be delivered
     */
    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mRgbaF = new Mat(height, width, CvType.CV_8UC4);
        mRgbaT = new Mat(width, width, CvType.CV_8UC4);
    }

    /**
     * This method is invoked when camera preview has been stopped for some reason.
     * No frames will be delivered via onCameraFrame() callback after this method is called.
     */
    @Override
    public void onCameraViewStopped() {
        mRgba.release();
    }

    /**
     * This method is invoked when delivery of the frame needs to be done.
     * The returned values - is a modified frame which needs to be displayed on the screen.
     * TODO: pass the parameters specifying the format of the frame (BPP, YUV or RGB and etc)
     *
     * @param inputFrame
     */
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        Core.transpose(mRgba, mRgbaT);
        Imgproc.resize(mRgbaT, mRgbaF, mRgbaF.size(), 0.0, 0, 0);
        Core.flip(mRgbaF, mRgba, 1);
        return mRgba;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case MY_PERMISSIONS_REQUEST_CAMERA: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Log.e("matti", "onRequestPermissionsResult: GRANTED");
                } else {
                    Log.e("matti", "onRequestPermissionsResult: DENIED");
                }
                break;
            }
            case MY_PERMISSION_REQUEST_WRITE: {
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Log.e("matti", "onRequestPermissionsResult: GRANTED");
                } else {
                    Log.e("matti", "onRequestPermissionsResult: DENIED");
                }
                break;

            }
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        //OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_6, this, mLoaderCallback);

        if (!OpenCVLoader.initDebug()) {
            Log.d("OpenCV", "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_6, this, mLoaderCallback);
        } else {
            Log.d("OpenCV", "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }

    }

/*
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
       // getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
*/

    /**
     * Activates camera intent
     */
    private void dispatchTakePictureIntent() {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (cameraIntent.resolveActivity(getPackageManager()) != null) {
            // Create the File where the photo should go
            File photoFile = null;
            photoFile = createImageFile();
            // Continue only if the File was successfully created
            if (photoFile != null) {

                Uri photoURI = FileProvider.getUriForFile(this, this.getApplicationContext().getPackageName() + ".provider", createImageFile());
                //cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, Uri.fromFile(photoFile));
                cameraIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(cameraIntent, REQUEST_IMAGE_CAPTURE);
            }
        }
    }

    /**
     * Creates the image file from the taken image
     */
    private File createImageFile() {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = this.getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = null;
        try {
            image = File.createTempFile(
                    imageFileName,  /* prefix */
                    ".jpg",         /* suffix */
                    storageDir      /* directory */
            );
        } catch (IOException e) {
            Log.e(TAG, "createImageFile: ", e);
        }

        // Save a file: path for use with ACTION_VIEW intents
        this.photoPath = image.getAbsolutePath();
        this.photoFile = image;
        return this.photoFile;
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {

/*
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == Activity.RESULT_OK) {
            Bitmap photo = (Bitmap) data.getExtras().get("data");
            Log.e(TAG, "onActivityResult: " + photo.getHeight() + " weight " + photo.getWidth() );
            mSrcImg.setImageBitmap(photo);
        }
*/

        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inSampleSize = 2;
            Bitmap imageBitmap = BitmapFactory.decodeFile(this.photoPath, options);
            if (imageBitmap != null) {
                try {
                    Log.e(TAG, "onActivityResult: photo file before write" + this.photoFile.length());
                    FileOutputStream fo = new FileOutputStream(this.photoFile);
                    imageBitmap.compress(Bitmap.CompressFormat.JPEG, 100, fo);
                    Log.e(TAG, "onActivityResult: photo file after write" + this.photoFile.length());
                } catch (FileNotFoundException e) {
                    Log.e(TAG, "onActivityResult: ", e);
                }

                imageBitmap = BitmapFactory.decodeFile(photoPath, options);

                if (imageBitmap != null)
                    Log.e(TAG, "onActivityResult: photo exists, size : " + imageBitmap.getByteCount());
                if (takingComparisonImage) {
                    mRefImg.setImageBitmap(imageBitmap);
                    comparisonImageChanged = true;
                } else {
                    mSrcImg.setImageBitmap(imageBitmap);
                }
                this.currentPhoto = imageBitmap;
            } else {
                Log.e(TAG, "onActivityResult: ERROR TAKING PICTURE");

            }
        }
    }

}