package com.lilium.shapedetection;

import com.github.sarxos.webcam.Webcam;
import com.lilium.shapedetection.util.ShapeDetectionUtil;
import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

class ShapeDetection {
    public static void main(String[] args) {
        OpenCV.loadLocally();
        final JPanel cameraFeed = new JPanel();
        final JPanel processedFeed = new JPanel();
        ShapeDetectionUtil.createJFrame(cameraFeed, processedFeed);
        Webcam webcam = Webcam.getDefault();
        webcam.setViewSize(new Dimension(640, 480)); // Set resolution
        webcam.open();
        startShapeDetection(cameraFeed, processedFeed, webcam).run();
    }

    private static Runnable startShapeDetection(final JPanel cameraFeed,
                                                final JPanel processedFeed,
                                                final Webcam webcam) {
        return () -> {
            while (true) {
                BufferedImage image = webcam.getImage();
                Mat frame = bufferedImageToMat(image);
                Mat adjustedFrame = adjustShadowsAndContrast(frame, 0.5, 1.2, 3.0);
                Mat processed = ShapeDetectionUtil.processImage(adjustedFrame);
                ShapeDetectionUtil.markOuterContour(processed, adjustedFrame);
                ShapeDetectionUtil.drawImage(adjustedFrame, cameraFeed);
                ShapeDetectionUtil.drawImage(processed, processedFeed);
                try {
                    Thread.sleep(33); // ~30 FPS
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        };
    }
    public static Mat adjustShadowsAndContrast(Mat frame, double shadowLevel, double contrastLevel, double sharpnessLevel) {
        Mat result = new Mat();
        frame.convertTo(result, -1, 1, shadowLevel);
        result.convertTo(result, -1, contrastLevel, 0);
        if (sharpnessLevel > 0) {
            Mat kernel = new Mat(3, 3, CvType.CV_32F);
            kernel.put(0, 0, 0, -1, 0, -1, 5 + sharpnessLevel, -1, 0, -1, 0); // Sharpening kernel
            Imgproc.filter2D(result, result, result.depth(), kernel);
        }

        return result;
    }

    private static Mat zoomImage(Mat input, double zoomFactor) {
        Mat zoomed = new Mat();
        int newWidth = (int) (input.cols() * zoomFactor);
        int newHeight = (int) (input.rows() * zoomFactor);
        Imgproc.resize(input, zoomed, new Size(newWidth, newHeight), 0, 0, Imgproc.INTER_LINEAR);
        return zoomed;
    }

    public static Mat bufferedImageToMat(BufferedImage bi) {
        int type = bi.getType();
        int channels;
        int cvType;

        switch (type) {
            case BufferedImage.TYPE_3BYTE_BGR:
                channels = 3;
                cvType = org.opencv.core.CvType.CV_8UC3;
                break;
            case BufferedImage.TYPE_BYTE_GRAY:
                channels = 1;
                cvType = org.opencv.core.CvType.CV_8UC1;
                break;
            default:
                BufferedImage bgrImage = new BufferedImage(bi.getWidth(), bi.getHeight(),
                        BufferedImage.TYPE_3BYTE_BGR);
                bgrImage.getGraphics().drawImage(bi, 0, 0, null);
                bi = bgrImage;
                channels = 3;
                cvType = org.opencv.core.CvType.CV_8UC3;
                break;
        }
        Mat mat = new Mat(bi.getHeight(), bi.getWidth(), cvType);
        byte[] data = new byte[bi.getWidth() * bi.getHeight() * channels];
        bi.getRaster().getDataElements(0, 0, bi.getWidth(), bi.getHeight(), data);
        if (channels == 3) {
            for (int i = 0; i < data.length; i += 3) {
                byte temp = data[i];
                data[i] = data[i + 2];
                data[i + 2] = temp;
            }
        }

        mat.put(0, 0, data);
        return mat;
    }
}
