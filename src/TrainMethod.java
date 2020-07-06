import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;
import static org.opencv.imgproc.Imgproc.INTER_LINEAR;
import static org.opencv.imgproc.Imgproc.getRotationMatrix2D;

public class TrainMethod {

    public static int index = 0;

    // 找到含有多张发票的多组二维码定位点集
    public void getLocationSet(Mat src, Point[][] pointSet) {
        // 灰度图
        Mat gray_image = new Mat();
        Imgproc.cvtColor(src, gray_image, Imgproc.COLOR_BGR2GRAY);

        // 阈值化
        Mat threshod_image = new Mat();
        Imgproc.threshold(gray_image, threshod_image, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);

        // 找出所有轮廓
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();  // 后、前、父、内四个轮廓索引
        Imgproc.findContours(threshod_image, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_NONE);

        /*
            1. 找出所有含有两个子轮廓的轮廓
            2. 符合要求的判断轮廓面积大小以及轮廓的形状
            3. 符合要求的放到temp_contours中

         */
        List<MatOfPoint> temp_contours = new ArrayList<>();

        int ic = 0;     // 将这个值加一传给其子轮廓，表示这是第几个子轮廓

        for (int i = 0; i < contours.size(); i++) {
            double[] temp_hierarchy = hierarchy.get(0, i);
            if (temp_hierarchy[2] != -1 && ic == 0) {
                ic++;
            } else if (temp_hierarchy[2] != -1) {
                ic++;
            } else if (temp_hierarchy[2] == -1) {
                ic = 0;
            }

            if (ic >= 2) {
                double area = Imgproc.contourArea(contours.get(i));

                if (area > 25 && area < 400) {
                    MatOfPoint2f temp_point2f = new MatOfPoint2f(contours.get(i).toArray());
                    RotatedRect rotatedRect = Imgproc.minAreaRect(temp_point2f);

                    double rect_width = rotatedRect.size.width;
                    double rect_height = rotatedRect.size.height;

                    double ratio = rect_width / rect_height;
                    if (ratio > 0.85 && ratio < 1.2) {
                        if (!(rect_width > 20 || rect_width < 5 || rect_height > 20 || rect_height < 5)) {
                            temp_contours.add(contours.get(i - 1));
                        }
                    }
                }
            }
        }

        /*
            消除不是二维码顶点的轮廓
            返回属于二维码顶点的点集

         */
        ClearNotQr(temp_contours, pointSet);

    }

    /*
        清除轮廓中非二维码顶点的轮廓，并且将二维码顶点分组存储到点集数组中

     */
    private void ClearNotQr(List<MatOfPoint> temp_contours, Point[][] pointSet) {
        int num = 0;

        // 依次扫描每个轮廓，找出其与其余轮廓的距离，构成等腰直角三角形的即为一个二维码
        for (int i = 0; i < temp_contours.size(); i++) {
            // 设置为空了的是被选出来作为二维码顶点的轮廓
            if (temp_contours.get(i) != null) {
                // 该轮廓的质心
                Point centerPoint = getPointLocation(temp_contours.get(i));

                double[] arr_length = new double[temp_contours.size()];
                for (int j = 0; j < temp_contours.size(); j++) {
                    if (temp_contours.get(j) != null) {
                        Point nextPoint = getPointLocation(temp_contours.get(j));

                        // 计算两个轮廓质心之间的距离，符合误差范围内的距离保存
                        double point_length = cal_length(centerPoint, nextPoint);

                        // 二维码边长不超过200 pix
                        if (point_length < 200 && point_length > 20) {
                            arr_length[j] = point_length;
                        } else {
                            arr_length[j] = 0;
                        }
                    }
                }

                // 找出arr_length数组中值相同的两个点
                for (int j = 0; j < arr_length.length; j++) {
                    if (arr_length[j] != 0) {
                        for (int k = j + 1; k < arr_length.length; k++) {
                            if (arr_length[k] != 0) {
                                // 误差百分之15
                                double tolerance = arr_length[j] * 0.15;
                                double deta = arr_length[j] - arr_length[k];
                                if (Math.abs(deta) < tolerance) {
                                    // 判断两线是否垂直
                                    // tan1 * tan2 = 1
                                    // centerPoint , j, k 三个点

                                    Point j_point = getPointLocation(temp_contours.get(j));
                                    Point k_point = getPointLocation(temp_contours.get(k));

                                    double tan1 = getTan(centerPoint, j_point);
                                    double tan2 = getTan(centerPoint, k_point);

                                    double tanadd = 0;
                                    if (Math.abs(tan1) > 10 || Math.abs(tan2) > 10) {
                                        if (Math.abs(tan1) < 0.1 || Math.abs(tan2) < 0.1) {
                                            tanadd = 1;
                                        }
                                    } else {
                                        tanadd = Math.abs(tan1 * tan2);
                                    }

                                    if (Math.abs(tanadd - 1) < 0.1) {
                                        // centerpoint j k 三个为二维码的三个顶点切 centerpoint为直角点
                                        // 判断二维码的朝向
                                        // 将这三个点从temp_contours中标记好去掉
                                        // 首先将这三个点加入到result中

                                        pointSet[num][0] = centerPoint;

                                        // 求j点绕center点顺时针旋转90度后的点x 的坐标
                                        double xx = centerPoint.x - (j_point.y - centerPoint.y);
                                        double xy = centerPoint.y + (j_point.x - centerPoint.x);

                                        Point xpoint = new Point(xx, xy);

                                        // 求 k , x 两点间的距离
                                        double xlength = cal_length(xpoint, k_point);

                                        if (xlength < tolerance) {
                                            // 即j为长轴
                                            pointSet[num][1] = j_point;
                                            pointSet[num][2] = k_point;
                                        } else {
                                            pointSet[num][1] = k_point;
                                            pointSet[num][2] = j_point;
                                        }

                                        temp_contours.set(i, null);
                                        temp_contours.set(j, null);
                                        temp_contours.set(k, null);
                                        num++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    }

    /*
        计算两点之间斜率

     */
    private double getTan(Point centerPoint, Point j_point) {
        double tan;
        double tanx = j_point.x - centerPoint.x;
        double tany = j_point.y - centerPoint.y;
        if (tanx == 0) {
            tan = 10000;
        } else {
            tan = tany / tanx;
        }
        return tan;
    }

    private double cal_length(Point centerPoint, Point nextPoint) {
        double point_length;
        double temp_x = centerPoint.x - nextPoint.x;
        double temp_y = centerPoint.y - nextPoint.y;
        point_length = Math.sqrt(Math.abs(temp_x * temp_x + temp_y * temp_y));

        return point_length;
    }

    private Point getPointLocation(MatOfPoint matOfPoint) {
        MatOfPoint2f point2f = new MatOfPoint2f(matOfPoint.toArray());
        RotatedRect rotatedRect = Imgproc.minAreaRect(point2f);

        // 该轮廓的质心
        return rotatedRect.center;
    }

    /*
        根据传入的二维码顶点集去分割发票

     */
    public void getTicket(Mat src, Point[][] pointSet) {
        for (Point[] points : pointSet) {
            if (points[0] != null) {
                // 根据二维码三个顶点计算出发票的四个顶点坐标
                Point[] rectPoint = getRect(points);

                // 根据四个顶点坐标计算出发票中心坐标，需要旋转的角度
                Point ticket_center = getPoint(rectPoint);
                double roll_angle = getAngle(rectPoint);
                System.out.println(roll_angle);

                // 求出发票的长宽和需要围绕中心点裁剪出来的正方形的边长
                double ticket_width = Math.sqrt(Math.abs((rectPoint[1].y - rectPoint[0].y) * (rectPoint[1].y - rectPoint[0].y) + (rectPoint[1].x - rectPoint[0].x) * (rectPoint[1].x - rectPoint[0].x)));
                double ticket_height = Math.sqrt(Math.abs((rectPoint[2].y - rectPoint[0].y) * (rectPoint[2].y - rectPoint[0].y) + (rectPoint[2].x - rectPoint[0].x) * (rectPoint[2].x - rectPoint[0].x)));
                int length = (int) Math.sqrt(ticket_height * ticket_height + ticket_width * ticket_width);

                // 创建一个正方形的空白mat
                Mat destImage = new Mat(length, length, src.type(), new Scalar(255, 255, 255));

                // 将发票的像素信息复制到空白的mat
                // double to int
                // 中心点坐标，length  int
                int pos_x = (int) ticket_center.x;
                int pos_y = (int) ticket_center.y;

//                Imgproc.circle(src, ticket_center, 10, new Scalar(0, 255, 0));


                // 先y后x   与  先x后y结果不一样
                // src.get(j,k)    j是row, k是col，  即j 是y， k是x


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
                Mat m = getRotationMatrix2D(centerp, roll_angle, 1.0);

                Imgproc.warpAffine(destImage, destImage, m, new Size(length, length), INTER_LINEAR, 0, new Scalar(255, 255, 255));


                double firstx = centerp.x - ticket_width / 2;
                double firsty = centerp.y - ticket_height / 2;
                double secondx = centerp.x + ticket_width / 2;
                double secondy = centerp.y + ticket_height / 2;

                Point firstPoint = new Point(firstx, firsty);
                Point secondPoint = new Point(secondx, secondy);

                Rect rect = new Rect(firstPoint, secondPoint);
                Mat newMat = new Mat(destImage, rect);

                String filename = "./images/trainFile" + index + ".png";
                index++;
                System.out.println(filename);
                Imgcodecs.imwrite(filename, newMat);
            }
        }
    }

    private double getAngle(Point[] rectPoint) {
        double deltay = rectPoint[0].y - rectPoint[1].y;
        double deltax = rectPoint[0].x - rectPoint[1].x;

        double rollAngle;

        if (deltax == 0) {
            // 竖直放
            if (deltay > 0) {
                rollAngle = -90;
            } else {
                rollAngle = 90;
            }
        } else {
            rollAngle = Math.toDegrees(Math.atan(deltay / deltax));
            if(deltax > 0){
                rollAngle = rollAngle + 180;
            }
        }
        return rollAngle;
    }

    private Point getPoint(Point[] rectPoint) {
        double maxx = 0;
        double maxy = 0;

        for (Point point : rectPoint) {
            maxx += point.x;
            maxy += point.y;
        }
        return new Point(maxx / 4, maxy / 4);
    }

    /*
        根据三个二维码顶点，求出发票的四个顶点

     */
    private Point[] getRect(Point[] points) {
        Point[] newPoint = new Point[4];
        double right_y = (points[1].y - points[0].y) * 1.6 + points[0].y;
        double right_x = (points[1].x - points[0].x) * 1.6 + points[0].x;

        double left_y = (points[0].y - points[1].y) * 7 + points[1].y;
        double left_x = (points[0].x - points[1].x) * 7 + points[1].x;

        double up_y = (points[0].y - points[2].y) * 4 + points[2].y;
        double up_x = (points[0].x - points[2].x) * 4 + points[2].x;

        double down_y = (points[2].y - points[0].y) * 1.7 + points[0].y;
        double down_x = (points[2].x - points[0].x) * 1.7 + points[0].x;


        // 根据上面四个顶点再求需要的矩形顶点
        double x1, x2, x3, x4;
        double y1, y2, y3, y4;

        if (up_x - down_x == 0) {
            x1 = left_x;
            y1 = up_y;

            x2 = right_x;
            y2 = up_y;

            x3 = left_x;
            y3 = down_y;

            x4 = right_x;
            y4 = down_y;

        } else if (right_x - left_x == 0) {
            x1 = up_x;
            y1 = left_y;

            x2 = up_x;
            y2 = right_y;

            x3 = down_x;
            y3 = left_y;

            x4 = down_x;
            y4 = right_y;
        } else {
            double k1 = (up_y - down_y) / (up_x - down_x);
            double k2 = (right_y - left_y) / (right_x - left_x);

            x1 = (k1 * left_x - left_y - k2 * up_x + up_y) / (k1 - k2);
            y1 = k1 * (x1 - left_x) + left_y;

            x2 = (k1 * right_x - right_y - k2 * up_x + up_y) / (k1 - k2);
            y2 = k1 * (x2 - right_x) + right_y;

            x3 = (k1 * left_x - left_y - k2 * down_x + down_y) / (k1 - k2);
            y3 = k1 * (x3 - left_x) + left_y;

            x4 = (k1 * right_x - right_y - k2 * down_x + down_y) / (k1 - k2);
            y4 = k1 * (x4 - right_x) + right_y;

        }

        newPoint[0] = new Point(x1, y1);
        newPoint[1] = new Point(x2, y2);
        newPoint[2] = new Point(x3, y3);
        newPoint[3] = new Point(x4, y4);

        return newPoint;
    }
}
