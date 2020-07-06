import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;

public class Taxi {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static final int LENGTH = 10;

    public static void main(String[] args) {
        try {
            Mat src = Imgcodecs.imread("./images/taxi.png");

            if (src.empty()) {
                throw new Exception("no image input");
            }

            TaxiMethod taxiMethod = new TaxiMethod();
            Point[] blackPoint = taxiMethod.getLocatePoint(src);

            Point[] rectPoint = new Point[LENGTH];
            Point[] elliPoint = new Point[LENGTH];
            taxiMethod.getTwoPointSet(src, rectPoint, elliPoint);

            Point[][] finalPoint = new Point[LENGTH][4];

            // 需要根据以上三个点集，找到一张发票上的矩形章和椭圆形章的位置
            for (int i = 0; i < rectPoint.length; i++) {
                if (rectPoint[i] != null) {
                    // 找到一组的三个点：两个定位黑块 和一个矩形章
                    finalPoint[i] = taxiMethod.getThreePoint(rectPoint[i], blackPoint);

                    // 通过上面找到的一组点，匹配找到第四个点
                    // 证明两条直线垂直   只需要证 tan 值
                    taxiMethod.getLastPoint(finalPoint[i], elliPoint);

                    // 通过两个章的位置裁剪出发票保存到文件中
                    taxiMethod.drawRect(src, finalPoint[i][0], finalPoint[i][3]);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
