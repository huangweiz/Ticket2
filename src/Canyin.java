import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgcodecs.Imgcodecs;

public class Canyin {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        try {
            Mat src = Imgcodecs.imread("./images/mul.png");

            if (src.empty()) {
                throw new Exception("no image input");
            }

            CanyinMethod canyinMethod = new CanyinMethod();
            Point[][] pointSet = new Point[20][3];

            canyinMethod.getLocationSet(src, pointSet);

            canyinMethod.getTicket(src, pointSet);

        } catch (Exception e) {
            e.printStackTrace();
        }


    }
}
