package com.samsung.shallolearning;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import com.samsung.shallolearning.ml.Elastic;

import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;


public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("elastic");
    }

    private static final int PERMISSION_REQUEST_CODE = 100;
    TextView txt_path , time_text , t_text;
    Button btn_file;
    Intent myFileIntent;
    String path;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Example of a call to a native method
        TextView tv = findViewById(R.id.sample_text);
        tv.setText("Hello world !");
        txt_path = (TextView) findViewById(R.id.textView);
        t_text = (TextView) findViewById(R.id.textView3);
        time_text = (TextView)findViewById(R.id.textView2);
        btn_file = (Button) findViewById(R.id.button);

        btn_file.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                String state = Environment.getExternalStorageState();
                if (Environment.MEDIA_MOUNTED.equals(state)) {
                    if (Build.VERSION.SDK_INT >= 23) {
                        if (checkPermission()) {
                            myFileIntent = new Intent(Intent.ACTION_GET_CONTENT);
                            myFileIntent.setType("*/*");
                            startActivityForResult(myFileIntent, 10);
                        } else {
                            requestPermission();
                        }
                    }
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        switch (requestCode) {
            case 10:
                if (resultCode == RESULT_OK) {
                    //start time for exeution

//                    path = data.getData().getPath();
//                    path = "/storage/emulated/0/Download/housing.csv";
                    path = "/storage/emulated/0/Download/car.csv";
//                    path = "/storage/emulated/0/Download/winequality-red.csv";
                    long objtptr = train(path);
                    double[] input = {-6.10949984e-01, -9.60095165e-01, -9.96573844e-01,
                            -1.06135241e+00, -7.20878455e-01, -6.75724541e-01,
                            -9.23077607e-01, -9.34823008e-01};
                    long start = System.currentTimeMillis();
                    txt_path.setText(test(objtptr , input) + "  rs is CPP prediction price");
                    long end = System.currentTimeMillis();
                    //end time for the execution
                    long t = end - start;
                    time_text.setText("Time Taken by cpp model " + t + " ms");

                    ByteBuffer byteBuffer = ByteBuffer.allocateDirect(8*4);
                    byteBuffer.putFloat(0, (float) -6.10949984e-01);
                    byteBuffer.putFloat(1 , (float) -9.60095165e-01);
                    byteBuffer.putFloat(2 , (float) -9.96573844e-01);
                    byteBuffer.putFloat(3, (float) -1.06135241e+00);
                    byteBuffer.putFloat(4 , (float) -7.20878455e-01);
                    byteBuffer.putFloat(5, (float) -6.75724541e-01);
                    byteBuffer.putFloat(6, (float) -9.23077607e-01);
                    byteBuffer.putFloat(7, (float) -9.34823008e-01);


                    try {
                        Elastic model = Elastic.newInstance(this);

                        // Creates inputs for reference.
                        long start1 = System.currentTimeMillis();
                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 8}, DataType.FLOAT32);
                        inputFeature0.loadBuffer(byteBuffer);

                        //start time for tflite predict api
                        // Runs model inference and gets result.
                        Elastic.Outputs outputs = model.process(inputFeature0);
                        long end1 = System.currentTimeMillis();
                        //end time for tflite predict api
                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                        float[] outputArray = outputFeature0.getFloatArray();
                        float a = outputArray[0];
                        t_text.setText("Time taken by tflite  "  + (end1 - start1)+ " ms");
                        // Releases model resources if no longer used.
                        model.close();
                    } catch (IOException e) {
                        t_text.setText("Some err");
                        // TODO Handle the exception
                    }

                }
        }
    }


    private boolean checkPermission() {
        int result = ContextCompat.checkSelfPermission(MainActivity.this, android.Manifest.permission.READ_EXTERNAL_STORAGE);
        if (result == PackageManager.PERMISSION_GRANTED) {
            return true;
        } else {
            return false;
        }
    }

    private void requestPermission() {
        if (ActivityCompat.shouldShowRequestPermissionRationale(MainActivity.this, android.Manifest.permission.READ_EXTERNAL_STORAGE)) {
            Toast.makeText(MainActivity.this, "Write External Storage permission allows us to read files. Please allow this permission in App Settings.", Toast.LENGTH_LONG).show();
        } else {
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{android.Manifest.permission.READ_EXTERNAL_STORAGE}, PERMISSION_REQUEST_CODE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[], int[] grantResults) {
        switch (requestCode) {
            case PERMISSION_REQUEST_CODE:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Log.e("value", "Permission Granted, Now you can use local drive .");
                } else {
                    Log.e("value", "Permission Denied, You cannot use local drive .");
                }
                break;
        }
    }
//    public static final int READ_EXTERNAL_STORAGE = 112;
//
//    protected void readSDcardDownloadedFiles() {
//        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
//                != PackageManager.PERMISSION_GRANTED) {
//            ActivityCompat.requestPermissions(this,
//                    new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, READ_EXTERNAL_STORAGE);
//        } else {
//            //Permission is granted
//            //Call the method to read file.
//        }
//    }
//
//    @Override
//    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
//        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
//        if (grantResults.length > 0
//                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
//            //Read the files
//        } else {
//            // permission denied, boo! Disable the
//            // functionality that depends on this permission.
//        }
//    }


    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native String test(long objptr , double[] inputs);
    public native long train(String path);

}