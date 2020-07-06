import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;

public class Train {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static final int MAXNUM = 20;

    public static void main(String[] args) {
        try {
            Mat src = Imgcodecs.imread("./images/ticket.png");

            if (src.empty()) {
                throw new Exception("no image input");
            }

            TrainMethod trainMethod = new TrainMethod();

            Point[][] pointSet = new Point[MAXNUM][3];

            trainMethod.getLocationSet(src, pointSet);

            // 根据二维码顶点集切割出发票
            trainMethod.getTicket(src, pointSet);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
