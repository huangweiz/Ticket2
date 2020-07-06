import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class TaxiMethod {
    public static final double MAXVALUE = 10;
    public static final double MINVALUE = 0.1;
    public static int index = 0;

    /*
        找到黑色定位块点集
        @input：含有多张车票的Mat 图
        @return：图片中所有黑色定位块的中心点集

     */
    public Point[] getLocatePoint(Mat src) {
        Point[] blackPoint = new Point[20];
        int num = 0;
        Mat gray = new Mat();
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);

        // 阈值大于某个值的时候设为0   即白色，否则不变
        // 找到两个定位点
        Imgproc.threshold(gray, gray, 60, 255, Imgproc.THRESH_BINARY);

        // 形态学操作加强两个定位点的形状
        // 矩形结构元素
        Mat k = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_RECT, new Size(3, 3));
        Imgproc.morphologyEx(gray, gray, Imgproc.MORPH_OPEN, k);
        Imgproc.morphologyEx(gray, gray, Imgproc.MORPH_CLOSE, k);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(gray, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area > 200 && area < 20000) {
                MatOfPoint2f temp_point2f = new MatOfPoint2f(contour.toArray());
                RotatedRect rotatedRect = Imgproc.minAreaRect(temp_point2f);

                blackPoint[num] = rotatedRect.center;
                num++;
            }
        }
        return blackPoint;
    }

    /*
        返回矩形章的位置和椭圆形章的位置
        @input: Mat 图， 矩形章点集， 椭圆形章点集
        根据红色来找出章的位置

     */
    public void getTwoPointSet(Mat src, Point[] rectPoint, Point[] elliPoint) {
        Mat hsv = new Mat();
        Imgproc.cvtColor(src, hsv, Imgproc.COLOR_BGR2HSV);

        int rectNum = 0;
        int elliNum = 0;

        // 只保留红色的部分
        Scalar lower_red = new Scalar(156, 43, 46);
        Scalar upper_red = new Scalar(180, 255, 255);

        Mat mask = new Mat();
        Core.inRange(hsv, lower_red, upper_red, mask);

        Mat res = new Mat();
        Core.bitwise_and(src, hsv, res, mask);

        Mat res_bin = new Mat();
        Imgproc.cvtColor(res, res_bin, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(res_bin, res_bin, 30, 255, Imgproc.THRESH_BINARY);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(res_bin, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);

            if (area > 1000) {
                MatOfPoint2f temp_point2f = new MatOfPoint2f(contour.toArray());
                RotatedRect rotatedRect = Imgproc.minAreaRect(temp_point2f);

                double rect_width = rotatedRect.size.width;
                double rect_height = rotatedRect.size.height;

                double ratio = rect_width / rect_height;
                double rectArea = rect_height * rect_width;
                double areaRatio = rectArea / area;

                if (ratio < 3 && ratio > (1 / 3)) {
                    if (areaRatio > 0.9 && areaRatio < 1.1) {
                        rectPoint[rectNum] = rotatedRect.center;
                        rectNum++;
                    } else {
                        elliPoint[elliNum] = rotatedRect.center;
                        elliNum++;
                    }
                }
            }
        }
    }

    /*
        根据矩形章和椭圆形章的两个位置切割出发票
        @input: Mat 图， 两个章的中心点

     */
    public void drawRect(Mat src, Point rect_center, Point elli_center) {
        double deltay = elli_center.y - rect_center.y;
        double deltax = elli_center.x - rect_center.x;

        double rollAngle = getAngle(rect_center, elli_center);

        double distance = Math.sqrt(Math.abs(deltax * deltax + deltay * deltay));

        double ticket_width = distance * 1.8;
        double ticket_height = distance * 5.8;
        int length = (int) Math.sqrt(ticket_width * ticket_width + ticket_height * ticket_height);

        Mat destImage = new Mat(length, length, src.type(), new Scalar(255, 255, 255));

        int pos_x = (int) rect_center.x;
        int pos_y = (int) rect_center.y;

        for (int j = (pos_y - length / 2), indexy = 0; j < (pos_y + length / 2); j++, indexy++) {
            if (j >= 0 && j <= src.rows()) {
                for (int k = (pos_x - length / 2), indexx = 0; k < (pos_x + length / 2); k++, indexx++) {
                    if (k >= 0 && k <= src.cols()) {
                        double[] tempdata = src.get(j, k);
                        if (tempdata != null) {
                            destImage.put(indexy, indexx, tempdata);
                        }
                    }
                }
            }
        }

        Point centerp = new Point(length / 2, length / 2);
        Mat m = Imgproc.getRotationMatrix2D(centerp, rollAngle, 1.0);

        Imgproc.warpAffine(destImage, destImage, m, new Size(length, length), Imgproc.INTER_LINEAR, 0, new Scalar(255, 255, 255));

        double leftx = centerp.x - distance * 0.9;
        double rightx = centerp.x + distance * 0.8;
        double uppery = centerp.y - distance * 1.6;
        double downy = centerp.y + distance * 2.9;

        Point point1 = new Point(leftx, uppery);
        Point point2 = new Point(rightx, downy);
        Rect rect = new Rect(point1, point2);

        Mat newMat = new Mat(destImage, rect);
        String filename = "./images/taxiFile" + index + ".png";
        index++;
        System.out.println(filename);
        Imgcodecs.imwrite(filename, newMat);

    }

    private double getAngle(Point rect_center, Point elli_center) {
        double deltay = elli_center.y - rect_center.y;
        double deltax = elli_center.x - rect_center.x;

        double rollAngle;

        if (deltay == 0) {
            // 水平放
            if (deltax < 0) {
                rollAngle = -90;
            } else {
                rollAngle = 90;
            }
        } else {
            rollAngle = -Math.toDegrees(Math.atan(deltax / deltay));
            // 如果deltay > 0 在旋转180度
            if (deltay > 0) {
                rollAngle = rollAngle + 180;
            }
        }
        return rollAngle;
    }


    /*
        根据一个矩形框的中心点，依次和所有定位黑块算距离，找出和它为同一张发票的两个黑色定位块的中心点
        根据三个点构成一个 最大角大于120度的 等腰直角三角形
        @input：矩形框中心点， 所有定位黑块中心点
        @return：point[], point[0] = 矩形框中心点， point[1],point[2]为同一组的黑色定位块中心点，
        point[3]为椭圆形章中心点，暂时为 null

     */
    public Point[] getThreePoint(Point point, Point[] blackPoint) {
        double[] all_length = new double[blackPoint.length];

        for (int i = 0; i < blackPoint.length; i++) {
            if (blackPoint[i] != null) {
                all_length[i] = point_length(point, blackPoint[i]);
            }
        }

        Point[] getPoint = new Point[4];

        aaa:
        for (int i = 0; i < blackPoint.length; i++) {
            if (all_length[i] != 0) {
                for (int j = i + 1; j < blackPoint.length; j++) {
                    if (all_length[j] != 0) {
                        double delta = Math.abs(all_length[i] - all_length[j]);
                        double ratio = delta / all_length[i];
                        if (ratio < 0.2) {
                            assert blackPoint[i] != null;
                            double temp_length = point_length(blackPoint[i], blackPoint[j]);
                            if (temp_length > all_length[i] * 1.5) {
                                // i,j 两个点匹配了该 章   置为0
                                getPoint[0] = point;
                                getPoint[1] = blackPoint[i];
                                getPoint[2] = blackPoint[j];
                                blackPoint[i] = null;
                                blackPoint[j] = null;
                                break aaa;
                            }
                        }
                    }

                }
            }
        }
        // 得到了匹配的三个点
        return getPoint;
    }

    /*
        两个点之间的距离

     */
    public double point_length(Point point1, Point point2) {
        double length;
        double deltax = point1.x - point2.x;
        double deltay = point1.y - point2.y;
        length = Math.sqrt(Math.abs(deltax * deltax + deltay * deltay));
        return length;
    }

    /*
        根据矩形章中心点和两个定位黑块中心点，在椭圆形章中心点集中找出同一组的椭圆形章中心点，放到point[3]中
        根据两个定位黑块中心点形成的直线和两个章形成的直线垂直

     */
    public void getLastPoint(Point[] points, Point[] elliPoint) {
        double k2x = points[1].x - points[2].x;
        double k2y = points[1].y - points[2].y;
        double k2;
        if (k2x == 0) k2 = MAXVALUE;
        else {
            k2 = k2y / k2x;
        }

        double k1x, k1y, k1;
        double tanadd = 0;

        for (Point point : elliPoint) {
            k1x = point.x - points[0].x;
            k1y = point.y - points[0].y;
            if (k1x == 0) {
                k1 = MAXVALUE;
            } else {
                k1 = k1y / k1x;
            }

            if (Math.abs(k1) >= MAXVALUE || Math.abs(k2) >= MAXVALUE) {
                if (Math.abs(k1) <= MINVALUE || Math.abs(k2) <= MINVALUE) {
                    tanadd = 1;
                }
            } else {
                tanadd = Math.abs(k1 * k2);
            }

            if (Math.abs(tanadd - 1) < MINVALUE) {
                points[3] = point;
                return;
            }
        }
    }
}
