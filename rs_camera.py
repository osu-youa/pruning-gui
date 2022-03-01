import pyrealsense2 as rs
import numpy as np

class Camera:
    def __init__(self, width, height, use_depth=True, fps=30):
        # Setup the camera

        self.width = width
        self.height = height
        self.use_depth = use_depth
        self.fps = fps
        self.last_intrinsics = None

        self.pc = rs.pointcloud()
        self.pipe = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        if self.use_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.align = rs.align(rs.stream.color)
        self.pipe.start(config)



    def acquire_image(self):
        frames = self.pipe.wait_for_frames()
        depth_img = None
        if self.use_depth:
            frames = self.align.process(frames)
            depth_img = np.asanyarray(frames.get_depth_frame().get_data())
        rgb_img = np.asanyarray(frames.get_color_frame().get_data())

        return rgb_img, depth_img

    def acquire_pc(self, return_rgb=False):
        if not self.use_depth:
            raise ValueError("Cannot obtain a PC when use_depth is False!")

        frames = self.pipe.wait_for_frames()
        aligned = self.align.process(frames)
        color_aligned_to_depth = aligned.first(rs.stream.color)
        depth_frame = frames.first(rs.stream.depth)
        points = self.pc.calculate(depth_frame)
        pt_array = np.asanyarray(points.get_vertices()).view(np.float32).reshape(self.height, self.width, 3)

        self.last_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        if not return_rgb:
            return pt_array

        rgb_array = np.asanyarray(color_aligned_to_depth.get_data())
        return pt_array, rgb_array

    def shutdown(self):
        self.pipe.stop()

    def deproject_pixel(self, pix, depth_scale=1.0):
        if self.last_intrinsics is None:
            raise Exception("No PC has been calculated so intrinsics have not been computed. Please compute a PC first")
        return np.array(rs.rs2_deproject_pixel_to_point(self.last_intrinsics, pix, depth_scale))



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from time import sleep

    cam = Camera(424, 240)

    # Testing point cloud generation
    pc, rgb = cam.acquire_pc(return_rgb=True)
    z_val = pc[:,:,2]
    plt.imshow(rgb)
    plt.show()
    plt.imshow(z_val)
    plt.show()


    # # Testing image capture
    # im_1, _ = cam.acquire_image()
    # print('Sleeping')
    # sleep(5.0)
    # im_2, _ = cam.acquire_image()
    #
    # plt.imshow(im_1)
    # plt.show()
    #
    # plt.imshow(im_2)
    # plt.show()




