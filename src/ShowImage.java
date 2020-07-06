import org.opencv.core.Mat;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

/*
    根据传入的Mat 多维矩阵对象，调用imageShow() 方法，将图片显示出来

 */

public class ShowImage {
    private JLabel imageView;
    private Mat mat;

    public ShowImage(Mat mat) {
        this.mat = mat;
        imageShow();
    }

    public void imageShow() {
        Image loadImage = toBufferedImage(mat);
        JFrame frame = createJFrame(mat.width(), mat.height());
        imageView.setIcon(new ImageIcon(loadImage));
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

    }

    private JFrame createJFrame(int width, int height) {
        JFrame frame = new JFrame();
        imageView = new JLabel();
        final JScrollPane imageScrollPane = new JScrollPane(imageView);
        imageScrollPane.setPreferredSize(new Dimension(width,height));
        frame.add(imageScrollPane,BorderLayout.CENTER);
        return frame;
    }

    private Image toBufferedImage(Mat mat) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (mat.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = mat.channels() * mat.cols() * mat.rows();
        byte[] buffer = new byte[bufferSize];
        mat.get(0, 0, buffer);

        // 获取所有的像素点
        BufferedImage image = new BufferedImage(mat.cols(), mat.rows(), type);
        final byte[] target = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(buffer, 0, target, 0, buffer.length);

        return image;
    }
}
