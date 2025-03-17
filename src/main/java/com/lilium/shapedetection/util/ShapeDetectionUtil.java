package com.lilium.shapedetection.util;

import org.opencv.core.Point;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Util class for shape detection.
 *
 * @author mirza
 */
public final class ShapeDetectionUtil {
    // region Constructor
    private ShapeDetectionUtil() {
        // Empty by default
    }
    // endregion

    // region OpenCV

    /**
     * Used to process forwarded {@link Mat} image and return the result.
     *
     * @param mat Image to process.
     * @return Returns processed image.
     */
    public static Mat processImage(final Mat mat) {
        final Mat processed = new Mat(mat.height(), mat.width(), mat.type());
        // Blur an image using a Gaussian filter
        Imgproc.bilateralFilter(mat, processed, 9, 75, 75);
        Imgproc.GaussianBlur(mat, processed, new Size(7, 7), 1);

        // Switch from RGB to GRAY
        Imgproc.cvtColor(processed, processed, Imgproc.COLOR_RGB2GRAY);

        // Find edges in an image using the Canny algorithm
        Imgproc.Canny(processed, processed, 200, 25);

        // Dilate an image by using a specific structuring element
        // https://en.wikipedia.org/wiki/Dilation_(morphology)
        Imgproc.dilate(processed, processed, new Mat(), new Point(-1, -1), 1);
        Imgproc.erode(processed, processed, new Mat());

        return processed;
    }

    public static void markOuterContour(final Mat processedImage, final Mat originalImage) {
        List<MatOfPoint> allContours = new ArrayList<>();
        Imgproc.findContours(processedImage, allContours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : allContours) {
            double area = Imgproc.contourArea(contour);
            if (area < 1000) continue; // Ignore small areas (noise)

            // Convert contour to MatOfPoint2f for rotated bounding box
            MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
            RotatedRect rotatedRect = Imgproc.minAreaRect(contour2f);

            Point[] rectPoints = new Point[4];
            rotatedRect.points(rectPoints);

            // Draw the rotated rectangle
            for (int i = 0; i < 4; i++) {
                Imgproc.line(originalImage, rectPoints[i], rectPoints[(i + 1) % 4], new Scalar(0, 255, 0), 2);
            }

            // Optionally, add text with size info
            String dimensions = findRotatedRectArea(rotatedRect);
            Imgproc.putText(originalImage, dimensions, rectPoints[0], Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 255), 1);
        }
    }



    private static String findRotatedRectArea(RotatedRect rotatedRect) {
        double scalingFactor = 0.0288;
        double lengthCm = rotatedRect.size.height * scalingFactor;
        double widthCm = rotatedRect.size.width * scalingFactor;
        return String.format("%.1f CM x %.1f CM", Math.max(lengthCm,widthCm), Math.min(lengthCm,widthCm));
    }


    private static String findArea(MatOfPoint contour) {
        Rect boundingRect = Imgproc.boundingRect(contour);
        double scalingFactor = 0.0359;
        double lengthCm = Math.round(boundingRect.height * scalingFactor);
        double widthCm = Math.round(boundingRect.width * scalingFactor);
        return String.format("l=%.2f cm, w=%.2f cm", lengthCm, widthCm);
    }

    public static void createJFrame(final JPanel... panels) {
        final JFrame window = new JFrame("Shape Detection");
        window.setSize(new Dimension(panels.length * 640, 480));
        window.setLocationRelativeTo(null);
        window.setResizable(false);
        window.setLayout(new GridLayout(1, panels.length));

        for (final JPanel panel : panels) {
            window.add(panel);
        }

        window.setVisible(true);
        window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    public static void drawImage(final Mat mat, final JPanel panel) {
        final BufferedImage image = ShapeDetectionUtil.convertMatToBufferedImage(mat);

        final Graphics graphics = panel.getGraphics();
        graphics.drawImage(image, 0, 0, panel);
    }

    private static BufferedImage convertMatToBufferedImage(final Mat mat) {
        final BufferedImage bufferedImage = new BufferedImage(
                mat.width(),
                mat.height(),
                mat.channels() == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_3BYTE_BGR
        );

        final WritableRaster raster = bufferedImage.getRaster();
        final DataBufferByte dataBuffer = (DataBufferByte) raster.getDataBuffer();
        mat.get(0, 0, dataBuffer.getData());

        return bufferedImage;
    }
}
