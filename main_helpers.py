from transmission.motors import Motor
from transmission.serial_conn import OurSerial

def get_motor_groups(JANK_CONTROLLER, speed_motor_channel, turn_motor_channel, weapon_motor_channel):
    # 5.1: Defining Transmission Object if we're using a live video
    ser = OurSerial()
    motor_group = Motor(ser=ser, channel=speed_motor_channel, channel2=turn_motor_channel)
    if JANK_CONTROLLER:
        weapon_motor_group = Motor(ser=ser, channel=weapon_motor_channel, speed=-1)
    else:
        weapon_motor_group = Motor(ser=ser, channel=weapon_motor_channel)
    return ser, motor_group, weapon_motor_group

def project_point_onto_line_segment(p, a, b):
    """
    Projects point p onto the line segment defined by points a and b.

    Args:
        p (tuple): The point to project (x_p, y_p).
        a (tuple): One endpoint of the line segment (x_a, y_a).
        b (tuple): The other endpoint of the line segment (x_b, y_b).

    Returns:
        tuple: The coordinates of the projected point (x, y) that lies on the segment.
    """
    # 1. Calculate the vector from A to B (v)
    v_x = b[0] - a[0]
    v_y = b[1] - a[1]
    
    # 2. Calculate the vector from A to P (w)
    w_x = p[0] - a[0]
    w_y = p[1] - a[1]
    
    # Calculate the squared length of the segment AB (len_sq_v)
    len_sq_v = v_x * v_x + v_y * v_y
    
    # Handle the case where A and B are the same point
    if len_sq_v == 0:
        return a # Or handle as an error/special case

    # 3. Calculate the dot product of vectors w and v (dot_wv)
    dot_wv = w_x * v_x + w_y * v_y
    
    # 4. Calculate the 't' parameter (projection factor)
    # t represents the normalized distance along the line AB where the projection lands
    t = dot_wv / len_sq_v
    
    # 5. Clamp 't' to the [0, 1] range to ensure the point stays on the SEGMENT
    # If t < 0, the projection is outside the segment on the A side, so return A
    if t < 0.0:
        return a
    # If t > 1, the projection is outside the segment on the B side, so return B
    elif t > 1.0:
        return b
    # If 0 <= t <= 1, the projection is ON the segment
    else:
        # Calculate the coordinates of the projected point
        projected_x = a[0] + t * v_x
        projected_y = a[1] + t * v_y
        return (projected_x, projected_y)