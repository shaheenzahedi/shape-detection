package com.lilium.shapedetection;

import com.github.sarxos.webcam.Webcam;
import com.lilium.shapedetection.util.ShapeDetectionUtil;
import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

class ShapeDetection {
    public static void main(String[] args) {
        OpenCV.loadLocally();
        // Create panels
        final JPanel cameraFeed = new JPanel();
        final JPanel processedFeed = new JPanel();
        ShapeDetectionUtil.createJFrame(cameraFeed, processedFeed);

        // Get default webcam
        Webcam webcam = Webcam.getDefault();
        webcam.setViewSize(new Dimension(640, 480)); // Set resolution
        webcam.open();

        // Start shape detection
        startShapeDetection(cameraFeed, processedFeed, webcam).run();
    }

    private static Runnable startShapeDetection(final JPanel cameraFeed,
                                                final JPanel processedFeed,
                                                final Webcam webcam) {
        return () -> {
            while (true) {
                BufferedImage image = webcam.getImage();
                Mat frame = bufferedImageToMat(image);
                double alpha = 2.0;
                int beta = -75;
                Mat adjustedFrame = new Mat();
                frame.convertTo(adjustedFrame, -1, alpha, beta);
//                Mat adjustedFrame = frame.clone();

                // Process frame (you'll need to convert BufferedImage to a format suitable for processing)
                Mat processed = ShapeDetectionUtil.processImage(adjustedFrame);

                // Mark outer contour
                ShapeDetectionUtil.markOuterContour(processed, adjustedFrame);
                // Draw current adjustedFrame
                ShapeDetectionUtil.drawImage(adjustedFrame, cameraFeed);

                // Draw processed image
                ShapeDetectionUtil.drawImage(processed, processedFeed);

                // Add small delay to prevent overwhelming the system
                try {
                    Thread.sleep(33); // ~30 FPS
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        };
    }

    private static Mat zoomImage(Mat input, double zoomFactor) {
        Mat zoomed = new Mat();
        int newWidth = (int) (input.cols() * zoomFactor);
        int newHeight = (int) (input.rows() * zoomFactor);
        Imgproc.resize(input, zoomed, new Size(newWidth, newHeight), 0, 0, Imgproc.INTER_LINEAR);
        return zoomed;
    }

    public static Mat bufferedImageToMat(BufferedImage bi) {
        // Determine the number of channels based on BufferedImage type
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
                // Convert to BGR if type is not directly supported
                BufferedImage bgrImage = new BufferedImage(bi.getWidth(), bi.getHeight(),
                        BufferedImage.TYPE_3BYTE_BGR);
                bgrImage.getGraphics().drawImage(bi, 0, 0, null);
                bi = bgrImage;
                channels = 3;
                cvType = org.opencv.core.CvType.CV_8UC3;
                break;
        }

        // Create Mat
        Mat mat = new Mat(bi.getHeight(), bi.getWidth(), cvType);

        // Get pixel data
        byte[] data = new byte[bi.getWidth() * bi.getHeight() * channels];
        bi.getRaster().getDataElements(0, 0, bi.getWidth(), bi.getHeight(), data);

        // If BGR, we need to reorder to RGB (OpenCV uses BGR by default)
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
