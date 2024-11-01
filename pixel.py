from manim import *

class PixelShuffle3DVisualization(ThreeDScene):
    def construct(self):
        # Define parameters
        input_grid_size = 2
        output_grid_size = 4
        channel_colors = [RED, GREEN, BLUE, YELLOW]

        # Create input cubes with 4 channels
        input_cubes = VGroup()
        for i in range(input_grid_size):
            for j in range(input_grid_size):
                cube = Cube(side_length=1)
                cube.move_to(np.array([j, -i, 0]))  # Place cubes in grid
                color_index = i * input_grid_size + j
                cube.set_fill(channel_colors[color_index], opacity=0.5)
                cube.set_stroke(color=WHITE, width=1.5)
                input_cubes.add(cube)

        # Position the input grid on the left side of the screen
        input_grid = VGroup(input_cubes).arrange_in_grid(rows=2, cols=2, buff=0.1).shift(LEFT * 3)
        input_label = Text("Input (2x2x4)", font_size=24).next_to(input_grid, UP)

        # Create output grid with 4x4 after pixel shuffle
        output_cubes = VGroup()
        for i in range(output_grid_size):
            for j in range(output_grid_size):
                cube = Cube(side_length=0.5)
                cube.move_to(np.array([j * 0.5, -i * 0.5, 0]))  # Place cubes in grid
                # Calculate color index based on pixel shuffle rearrangement
                color_index = (i % 2) * 2 + (j % 2)  
                cube.set_fill(channel_colors[color_index], opacity=0.5)
                cube.set_stroke(color=WHITE, width=1)
                output_cubes.add(cube)

        # Position the output grid on the right side of the screen
        output_grid = VGroup(output_cubes).arrange_in_grid(rows=4, cols=4, buff=0.05).shift(RIGHT * 3)
        output_label = Text("Output (4x4)", font_size=24).next_to(output_grid, UP)

        # Set up the 3D camera orientation
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)

        # Add input and output grids with labels to the scene
        self.play(FadeIn(input_grid), Write(input_label))
        self.play(FadeIn(output_grid), Write(output_label))

        # Animation to transform input grid into output grid
        self.wait(1)
        self.play(
            *[Transform(input_cubes[i], output_cubes[i * 4:(i + 1) * 4]) for i in range(4)],
            run_time=2
        )
        self.wait(1)

        # Move the camera to create a rotation effect
        self.play(self.camera.set_euler_angles(theta=PI/4), run_time=2)
        self.wait(1)

