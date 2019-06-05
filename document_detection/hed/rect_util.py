import numpy as np
import cv2
import math

koughLinesPThreshold = 200
kHoughLinesPMinLinLength = 100
kHoughLinesPMaxLineGap = 50

kIntersectionMinAngle = 45
kIntersectionMaxAngle = 135
kCloserPointMaxDistance = 6.0
kPointOnLineMaxOffset = 8
kSameSegmentsMaxAngle = 5
kMergeLinesMaxDistance = 5
kRectOpposingSidesMinRatio = 0.5


class Point:
    def __init__(self, x_init, y_init):
        self.x = x_init
        self.y = y_init

    def shift(self, x, y):
        self.x += x
        self.y += y

    def __repr__(self):
        return "".join(["Point(", str(self.x), ",", str(self.y), ")"])


class Corner:
    def __init__(self):
        self.point = None
        self.segments = []


def _get_angle_of_line(line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    # http://stackoverflow.com/questions/1311049/how-to-map-atan2-to-degrees-0-360
    # degrees = (degrees + 360) % 360 这种修正办法，得到的是 int 类型的角度，虽然损失了一点点精度，但是还是可以满足这里算法的需求
    angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
    fix_angle = (int(angle) + 360) % 360
    return fix_angle


def _get_ref_line(line, image_width, image_height):
    """
     line format is (x_{start}, y_{start}, x_{end}, y_{end})
     分别对应          line[0]    line[1]    line[2]  line[3]

     公式为 y = a*x + b

     根据下面两个等式
     line[1] = a*line[0] + b
     line[3] = a*line[2] + b

     可以进行推导
     line[1] - line[3] = a* (line[0] - line[2])

     得到
     a = (line[1] - line[3]) / (line[0] - line[2])
     b = line[1] - a*line[0]
       = (line[0]*line[3] - line[2]*line[1]) / (line[0] - line[2])
    """
    ref_line = [None, None, None, None]
    if line[0] == line[2]:
        # 和 Y 轴平行的线，按照从下往上的方向排列
        ref_line[0] = line[0]
        ref_line[1] = 0  # 从下往上
        ref_line[2] = line[2]
        ref_line[3] = image_height
    elif line[1] == line[3]:
        # 和 X 轴平行的线，按照从左往右的方向
        ref_line[0] = 0  # 从左往右
        ref_line[1] = line[1]
        ref_line[2] = image_width
        ref_line[3] = line[3]
    else:
        # 这个分支的斜线才能通过公式进行计算，而且不会遇到下面这个除法中(line[0] - line[2]) == 0 的情况，避免计算错误
        # a = (line[1] - line[3]) / (line[0] - line[2])
        a = (line[1] - line[3]) / (line[0] - line[2])
        b = (line[0] * line[3] - line[2] * line[1]) / (line[0] - line[2])

        # y = a * x + b
        ref_line[0] = 0  # 从左往右
        ref_line[1] = int(b)
        ref_line[2] = int((image_height - b) / a)
        ref_line[3] = image_height  # ref_line[3] = a * ref_line[2] + b

    return ref_line


def _is_two_ref_line_close_to_each_other(line_a, line_b):
    if abs(line_a[1] - line_b[1]) < kMergeLinesMaxDistance and abs(line_a[3] - line_b[3]) < kMergeLinesMaxDistance:
        return True
    return False


def _merge_ref_line_and_segment_pairs(ref_line_and_segment_pairs, image_width, image_height):
    """
    HoughLinesP检测出来的是线段，是有长度的。
    把每一个线段扩展成一个统一规格的 RefLineVec4i，形成一个 pair，然后调用这个函数，对这些 pair 进行合并。
    用RefLineVec4i来决策是否需要合并，对于需要合并的，则把对应的HoughLinesP线段重组成一个更长的线段。
    """
    merged_ref_line_and_segment_pairs = []
    for i in range(len(ref_line_and_segment_pairs)):
        ref_line_and_segment = ref_line_and_segment_pairs[i]
        ref_line = ref_line_and_segment[0]
        segment = ref_line_and_segment[1]

        if len(merged_ref_line_and_segment_pairs) == 0:
            merged_ref_line_and_segment_pairs.append((ref_line, segment))
        else:
            is_closer = False
            for j in range(len(merged_ref_line_and_segment_pairs)):
                merged_ref_line_and_segment = merged_ref_line_and_segment_pairs[j]
                merged_ref_line = merged_ref_line_and_segment[0]
                merged_segment = merged_ref_line_and_segment[1]

                if _is_two_ref_line_close_to_each_other(ref_line, merged_ref_line):
                    # 如果两条 ref line 很接近，则把两个segment合并成一个，然后重新生成新的 ref line
                    # 先取出4个点
                    point_vector = [Point(segment[0], segment[1]), Point(segment[2], segment[3]),
                                    Point(merged_segment[0], merged_segment[1]),
                                    Point(merged_segment[2], merged_segment[3])]

                    # 排序之后，得到最左边和最右边的两个 point，这两个 point 就可以构成新的线段
                    sorted(point_vector, key=lambda k: point_vector[k].point.x)
                    left_most_point = point_vector[0]
                    right_most_point = point_vector[3]

                    new_segment = []
                    new_segment[0] = left_most_point.x
                    new_segment[1] = left_most_point.y
                    new_segment[2] = right_most_point.x
                    new_segment[3] = right_most_point.y

                    # TODO，考虑一下，这里是否需要使用其他的线段合并策略，是否需要把新的线段的两个 point，做一个细微调整，
                    #  让这两个 point 正好处于新的直线上

                    new_ref_line = _get_ref_line(new_segment, image_width, image_height)
                    merged_ref_line_and_segment_pairs[j] = (new_ref_line, new_segment)
                    is_closer = True
                    break

            if not is_closer:
                merged_ref_line_and_segment_pairs.append((ref_line, segment))

    return merged_ref_line_and_segment_pairs


def _cross(pa, pb):
    result = [float(pa[1]) * pb[2] - float(pa[2]) * pb[1], float(pa[2]) * pb[0] - float(pa[0]) * pb[2],
              float(pa[0]) * pb[1] - float(pa[1]) * pb[0]]
    return result


def _is_point_on_line(point, line):
    p0 = Point(line[0], line[1])
    p1 = Point(line[2], line[3])

    # 在HED和霍夫曼检测的时候，矩形的拐角处的两条线段，可能会断开，所以这里在line的两端，适当的延长一点点距离
    min_x = min(p0.x, p1.x) - kPointOnLineMaxOffset
    max_x = max(p0.x, p1.x) + kPointOnLineMaxOffset
    min_y = min(p0.y, p1.y) - kPointOnLineMaxOffset
    max_y = max(p0.y, p1.y) + kPointOnLineMaxOffset

    if min_x <= point.x <= max_x and min_y <= point.y <= max_y:
        return True
    return False


def _get_segment_intersection(line_a, line_b):
    """
    这个版本，line实际上是有限长度的线段，所以还额外检测了一下 point 是否在线段上
    """
    pa = [None, None, None]
    pa[0] = line_a[0]
    pa[1] = line_a[1]
    pa[2] = 1

    pb = [None, None, None]
    pb[0] = line_a[2]
    pb[1] = line_a[3]
    pb[2] = 1

    la = _cross(pa, pb)

    pa[0] = line_b[0]
    pa[1] = line_b[1]
    pa[2] = 1

    pb[0] = line_b[2]
    pb[1] = line_b[3]
    pb[2] = 1

    lb = _cross(pa, pb)

    inter = _cross(la, lb)

    if inter[2] == 0:
        return None, False  # two lines are parallel
    else:
        intersection = Point(inter[0] / inter[2], inter[1] / inter[2])
        if _is_point_on_line(intersection, line_a) and _is_point_on_line(intersection, line_b):
            return intersection, True
        return None, False


def _arrange_rect_corners(rect_corners):
    """
    按照顺时针排序，对4个 corner 排序，得到 4 corners:
        top-left, top-right, bottom-right, bottom-left, index are 0, 1, 2, 3
    :param rect_corners:
    :return:
    """
    sorted(rect_corners, key=lambda k: left_two_corners[k].point.x)

    left_two_corners = []
    right_two_corners = []
    left_two_corners.append(rect_corners[0])
    left_two_corners.append(rect_corners[1])
    right_two_corners.append(rect_corners[2])
    right_two_corners.append(rect_corners[3])

    sorted(left_two_corners, key=lambda k: left_two_corners[k].point.y)
    sorted(right_two_corners, key=lambda k: right_two_corners[k].point.y)

    sorted_corners = [left_two_corners[0], right_two_corners[0], right_two_corners[1], right_two_corners[0]]
    return sorted_corners


def _is_segments_has_same_segment(segments, segment, image_width):
    for seg in segments:
        angle_a = _get_angle_of_line(seg)
        angle_b = _get_angle_of_line(segment)

        diff = abs(angle_a - angle_b)
        diff = diff % 90  # 修正到0~90度

        if diff < kSameSegmentsMaxAngle or diff > (90 - kSameSegmentsMaxAngle):
            return True
    return False


def _point_distance(a, b):
    x_distance = a.x - b.x
    y_distance = a.y - b.y
    distance = math.sqrt(x_distance * x_distance + y_distance * y_distance)
    return distance


def _get_angle_of_two_points(point_a, point_b):
    angle = math.atan2(point_b.y - point_a.y, point_b.x - point_a.x) * 180.0 / np.pi
    fix_angle = (int(angle) + 360) % 360
    return fix_angle


def _is_rect_corners_reasonable(rect_corners, image_width):
    """
    一组策略，判断4个 corner 是否可以形成一个可信度高的矩形(有透视变换效果，所以肯定不是标准的长方形，而是一个梯形或平行四边形)
     4个 point 是已经经过ArrangeRectPoints排过序的
     4 points top-left, top-right, bottom-right, bottom-left, index are 0, 1, 2, 3
    """
    rect_points = [rect_corners[0].point,
                   rect_corners[1].point,
                   rect_corners[2].point,
                   rect_corners[3].point]

    segment_0_to_1 = [rect_points[0].x, rect_points[0].y, rect_points[1].x, rect_points[1].y]
    segment_1_to_2 = [rect_points[1].x, rect_points[1].y, rect_points[2].x, rect_points[2].y]
    segment_2_to_3 = [rect_points[2].x, rect_points[2].y, rect_points[3].x, rect_points[3].y]
    segment_3_to_0 = [rect_points[3].x, rect_points[3].y, rect_points[0].x, rect_points[0].y]

    rect_segments = [segment_0_to_1, segment_1_to_2, segment_2_to_3, segment_3_to_0]

    # segment_0_to_1这条线段，应该和rect_corners[0]的所有 segments 里面的至少一条线段是相似的，同时，
    # segment_0_to_1这条线段，也应该和rect_corners[1]的所有 segments 里面的至少一条线段是相似的

    if not (_is_segments_has_same_segment(rect_corners[0].segments, segment_0_to_1, image_width) and
            _is_segments_has_same_segment(rect_corners[1].segments, segment_0_to_1, image_width)):
        return False

    if not (_is_segments_has_same_segment(rect_corners[1].segments, segment_1_to_2, image_width) and
            _is_segments_has_same_segment(rect_corners[2].segments, segment_1_to_2, image_width)):
        return False

    if not (_is_segments_has_same_segment(rect_corners[2].segments, segment_2_to_3, image_width) and
            _is_segments_has_same_segment(rect_corners[3].segments, segment_2_to_3, image_width)):
        return False

    if not (_is_segments_has_same_segment(rect_corners[3].segments, segment_3_to_0, image_width) and
            _is_segments_has_same_segment(rect_corners[0].segments, segment_3_to_0, image_width)):
        return False

    # 第二组策略，根据四边形的形状
    distance_of_0_to_1 = _point_distance(rect_points[0], rect_points[1])
    distance_of_1_to_2 = _point_distance(rect_points[1], rect_points[2])
    distance_of_2_to_3 = _point_distance(rect_points[2], rect_points[3])
    distance_of_3_to_0 = _point_distance(rect_points[3], rect_points[0])

    # 计算两组对边的比例(0.0 -- 1.0的值)
    # 两条对边(标准矩形的时候，就是两条平行边)的 minLength / maxLength，不能小于0.5，
    # 否则就认为不是矩形(本来是把这个阈值设置为0.8的，但是因为图片都是缩放后进行的处理，
    # 长宽比有很大的变化，所以把这里的过滤条件放宽一些，设置为0.5)
    # distance_of_0_to_1 和 distance_of_2_to_3 是两条对边
    ratio1 = min(distance_of_0_to_1, distance_of_2_to_3) / max(distance_of_0_to_1, distance_of_2_to_3)
    ratio2 = min(distance_of_1_to_2, distance_of_3_to_0) / max(distance_of_1_to_2, distance_of_3_to_0)

    if ratio1 >= kRectOpposingSidesMinRatio and ratio2 >= kRectOpposingSidesMinRatio:
        # 两组对边，至少有一组是接近平行状态的(根据透视角度的不同，四边形是一个梯形或者平行四边形) 用这个再做一轮判断
        # 4 条边和水平轴的夹角
        angle_top = _get_angle_of_line(rect_points[1], rect_points[0])
        angle_bottom = _get_angle_of_two_points(rect_points[2], rect_points[3])
        angle_right = _get_angle_of_two_points(rect_points[2], rect_points[1])
        angle_left = _get_angle_of_two_points(rect_points[3], rect_points[0])

        diff1 = abs(angle_top - angle_bottom)
        diff2 = abs(angle_right - angle_left)
        diff1 = diff1 % 90
        diff2 = diff2 % 90  # 修正到0~90度

        # 这里的几个值，都是经验值
        if diff1 <= 8 and diff2 <= 8:
            # 俯视拍摄，平行四边形
            return True

        if diff1 <= 8 and diff2 <= 45:
            # 梯形，有透视效果
            return True

        if diff1 <= 45 and diff2 <= 8:
            # 梯形，有透视效果
            return True

    return False


def find_rect_procesor(img_path):
    img_o = cv2.imread(img_path)
    (bg_h, bg_w, _) = img_o.shape
    img_l = np.zeros((bg_h, bg_w, 1), np.uint8)
    img_p = np.zeros((bg_h, bg_w, 1), np.uint8)
    img = cv2.imread(img_path, 0)

    # 找线段
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, koughLinesPThreshold, kHoughLinesPMinLinLength, kHoughLinesPMaxLineGap)
    lines1 = lines[:, 0, :]
    for x1, y1, x2, y2 in lines1[:]:
        # cv2.line(img_o, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(img_p, (x1, y1), (x2, y2), 255, 2)

    # 线段转换成 参考直线(其实是正好被 image 完整尺寸截断的线段)，并且做一轮过滤
    ref_line_and_segment_pairs = []
    for segment in lines:
        segment = segment[0]
        print(segment)
        ref_line = _get_ref_line(segment, bg_w, bg_h)

        # 线段长度过滤
        segment_length = math.sqrt(
            (segment[1] - segment[3]) * (segment[1] - segment[3]) + (segment[0] - segment[2]) * (
                    segment[0] - segment[2]))

        if segment_length > kHoughLinesPMinLinLength:
            ref_line_and_segment_pairs.append((ref_line, segment))

    # 合并临近的直线
    ref_lines = []
    segments = []
    # merged_ref_line_and_segment_pairs = _merge_ref_line_and_segment_pairs(ref_line_and_segment_pairs, bg_w, bg_h)
    for ref_line, segment in ref_line_and_segment_pairs:
        ref_lines.append(ref_line)
        segments.append(segment)

    # 寻找segment线段的交叉点以及过滤
    all_corners = []
    corners = []
    for i in range(len(segments)):
        for j in range(len(segments)):
            segment_a = segments[i]
            segment_b = segments[j]
            # https://gist.github.com/ceykmc/18d3f82aaa174098f145 two lines intersection
            # http://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python
            intersection_point, state = _get_segment_intersection(segment_a, segment_b)
            if state:
                all_corners.append(intersection_point)
                # 对交叉点进行第一轮过滤
                if intersection_point.x <= 0 or intersection_point.y <= 0 or intersection_point.x > bg_w or intersection_point.y > bg_h:
                    # 交叉点如果在图片外部，也需要过滤掉
                    pass
                else:
                    theta_a = _get_angle_of_line(segment_a)
                    theta_b = _get_angle_of_line(segment_b)

                    angle = abs(theta_a - theta_b)
                    angle = angle % 180  # 再修正到180度范围内
                    if kIntersectionMinAngle <= angle <= kIntersectionMaxAngle:
                        # 基于两条线的角度进行过滤
                        t_c = Corner()
                        t_c.point = intersection_point
                        t_c.segments.append(segment_a)
                        t_c.segments.append(segment_b)
                        corners.append(t_c)

    # 对交叉点进行第二轮过滤，两个点如果很接近，则合并成同一个点，并且用他们的平均值来标示该点
    average_corners = []
    for i in range(len(corners)):
        corner = corners[i]
        if len(average_corners) == 0:
            average_corners.append(corner)
        else:
            is_closer = False
            for j in range(len(average_corners)):
                c = average_corners[j]
                # TODO 减法重载
                new_cornet = Corner()
                diff = corner.point - c.point
                distance = diff.x * diff.x + diff.y * diff.y
                if distance < kCloserPointMaxDistance:
                    # 两个点很靠近，合并成同一个点
                    new_point = Point((corner.point.x + c.point.x) / 2, (corner.point.y + c.point.y) / 2)
                    new_cornet.point = new_point

                    # 还要合并每个 cornet 的 segment 线段数组
                    new_cornet.segments = corner.segments + c.segments

                    average_corners[j] = new_cornet
                    isCloser = True
                    break

            if not is_closer:
                average_corners.append(corner)

    # 寻找四边形
    if len(average_corners) >= 4:
        # 至少要有4个点，才算是矩形
        # TODO，如果点太多，还会影响计算性能，所以可能还需要一个上限值，并且，当达到上限值的时候，还需要考虑如何进一步处理，减少点的数量
        max_perimeter = 0.0

        rect_corners = []
        rect_corners_with_max_perimeter = []
        rect_points_with_max_perimeter = []
        for i in range(len(average_corners) - 4):
            for j in range(i + 1, len(average_corners) - 3):
                for m in range(j + 1, len(average_corners) - 2):
                    for n in range(m + 1, len(average_corners) - 1):
                        rect_corners = [average_corners[i], average_corners[j], average_corners[m], average_corners[n]]

                        # 对四个点按照顺时针方向排序
                        rect_corners = _arrange_rect_corners(rect_corners)

                        # 如果不是一个合理的四边形，则直接排除
                        if _is_rect_corners_reasonable(rect_corners, bg_w) == False:
                            continue

                        rect_points = [rect_corners[0].point, rect_corners[1].point, rect_corners[2].point,
                                       rect_corners[3].point]

                        perimeter = 0
                        # perimeter = contourArea(rect_points); #  或者用最大面积
                        # perimeter = arcLength(rect_points, true); # 最大周长
                        if perimeter > max_perimeter:
                            max_perimeter = perimeter
                            rect_corners_with_max_perimeter = rect_corners
                            rect_points_with_max_perimeter = rect_points

        if len(rect_points_with_max_perimeter) == 4:
            results = rect_points_with_max_perimeter

    find_rect = len(results) == 4
    return (find_rect, results)


if __name__ == "__main__":
    print(find_rect_procesor('./out/bJW2weiuLJ_ORIGINAL.jpg'))
