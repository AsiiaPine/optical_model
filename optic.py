from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class BarrierPosition:
    '''
    Position relative to the Optopair LED
    '''
    y: float
    z: float
    width : float
    def __init__(self, y, z, width):
        self.y = y
        self.z = z
        self.width = width

class OptoPairParameters:
    def __init__(self, y_shift=0, x_shift = 0.4,
            # FROM article
            l0 = 3.5, O_ff = 0, w = 0.1,

            # sides of Infrared LED
            S1  = 0.35 , S2 = 0.35 ,

            # sides of LED metal platform (approx)
            dps1 = 0.1, dps2 = 0.1,
            # sides of phototransistor
            X = 0.22,Y = 0.76, 
            dpx = 0.16, dpy = 0.29,
            n_led_rows = 5,
            n_led_columns = 5,
            n_pt_rows = 38,
            n_pt_columns = 11
        ):
        
        self.y_shift = y_shift
        self.x_shift = x_shift
        self.l0 = l0
        self.O_ff = O_ff
        self.w = w

        # sides of LED metal platform (approx)
        self.dps1 = dps1
        self.dps2 = dps2

        # sides of PT metal platform (approx)
        self.dpx = dpx
        self.dpx = dpy
        self.pt_blind_shape = 8, 15
        self.led_blind_shape = n_led_rows * 3 // 5, n_led_columns * 3 // 5
        self.pt_blind_x = 8
        self.pt_blind_y = 15
        self.led = OptoParameters(S1, S2, n_led_rows, n_led_columns, self.led_blind_shape)
        self.pt = OptoParameters(X, Y, n_pt_rows, n_pt_columns, self.pt_blind_shape)


class OptoParameters:
    def __init__(self, X = 0.35, Y = 0.35, n_rows = 5, n_columns = 5, blind_shape = None):
        self.X = X
        self.Y = Y
        self.n_rows = n_rows
        self.n_columns = n_columns
        
        # Matrix of metal plate inside LED
        self.blind_shape = blind_shape

class Optic:
    def __init__(self, parameters : OptoParameters, is_metal_plate_centered = True):
        self.param = parameters
        self.blindness_matrix = make_blindness_matrix(
            (parameters.n_rows, parameters.n_columns),
            parameters.blind_shape, is_metal_plate_centered)
        self.matrix = divide_surface(
            (parameters.X, parameters.Y), 
            parameters.n_rows, parameters.n_columns)
        self.intensity_matrix = np.zeros(shape=( parameters.n_rows, parameters.n_columns))

class OptoPair:
    def __init__(self, params: OptoPairParameters,):
        self.params = params

        self.led_parameters = params.led
        self.led = Optic(self.led_parameters)

        self.led.matrix[:, :, 0] -= self.led_parameters.X/2
        self.led.matrix[:, :, 1] -= self.led_parameters.Y/2

        x_shift = params.x_shift
        y_shift = params.y_shift
        
        self.pt_parameters = params.pt
        self.pt_mes = Optic(self.pt_parameters, False)

        self.pt_mes.matrix[:, :, 0] += x_shift - self.pt_parameters.X/2
        self.pt_mes.matrix[:, :, 1] -= y_shift + self.pt_parameters.Y/2
        self.pt_ref = Optic(self.pt_parameters, False)

        self.pt_ref.matrix[:, :, 0] -= x_shift + self.pt_parameters.X/2
        self.pt_ref.matrix[:, :, 1] -= y_shift + self.pt_parameters.Y/2

        self.pt_ref.blindness_matrix = self.recalculate_pt_blindness(self.pt_ref.blindness_matrix)
        self.pt_mes.blindness_matrix = self.recalculate_pt_blindness(self.pt_mes.blindness_matrix)

        self.pt_mes.intensity_matrix = self.calculate_intensity()
        self.pt_ref.intensity_matrix = self.calculate_intensity(False)

    def calculate_intensity(self, for_mes = True):
        PT_matrix = None
        pt: Optic
        LED_matrix = self.led.matrix
        LED_blindness_matrix = self.led.blindness_matrix
        if for_mes:
            pt = self.pt_mes
            PT_matrix = self.pt_mes.matrix
        else: 
            pt = self.pt_ref
            PT_matrix = self.pt_ref.matrix
            
        pt_blind_x, pt_blind_y = self.pt_parameters.blind_shape
        
        PT_intensity = np.zeros(shape=(pt.param.n_rows, pt.param.n_columns))
        
        for i in np.arange(PT_matrix.shape[0]):
            for j in np.arange(PT_matrix.shape[1]):
                if pt_blind_x > i or pt_blind_y > j:
                    PT_intensity[i, j] =  calc_intensity_matrix_to_point(PT_matrix[i, j],
                                                                         LED_matrix, 
                                                                         led_map=LED_blindness_matrix, 
                                                                         l_0=-1, I_0=0)
        return PT_intensity
    
    def recalculate_pt_blindness(self, pt_blindness, pt_blind_x = 8, pt_blind_y = 15):
        '''
            pt_blind_x and pt_blind_y defines 
            the area of the PT metallic plate in cells of the optic element
        '''
        
        pt_blind_x, pt_blind_y = self.pt_parameters.blind_shape
        for i in range(pt_blindness.shape[0]):
            for j in range(pt_blindness.shape[1]):
                if pt_blind_x>=j and pt_blind_y >= i:
                    pt_blindness[i, j] = 0
        return pt_blindness
    
    @property
    def parameters(self):
        return self.params

    def get_intensity(self, barrier=None):
        PT_matrix_intensity_ref = self.calculate_intensity(False)

        PT_intensity_barr = None
        if isinstance(barrier, BarrierPosition):
            PT_intensity_barr = self.calculate_intensity_with_barrier(barrier=barrier)
        else: PT_intensity_barr = self.calculate_intensity()
        
        return np.sum(PT_intensity_barr)/np.sum(PT_matrix_intensity_ref)
        
    def calculate_extreme_ys_for_pt(self, barrier: BarrierPosition):
        l0 = self.params.l0
        pt_ys = self.pt_mes.matrix[:, 1][:, 1]
        extreme_ys = np.zeros_like(pt_ys)
        y = barrier.y
        z = barrier.z
        width = barrier.width
        
        y = barrier.y
        for i, extreme_y in enumerate(extreme_ys):
            y_pt = pt_ys[i]

            dy = y - y_pt
            dz = z
            if (y_pt > y):
                dz += (width / 2)
            else:
                dz -= (width / 2)

            extreme_ys[i] = dy * l0 / dz
        return extreme_ys

    def calculate_intensity_with_barrier(self, barrier: BarrierPosition):
        extreme_ys = self.calculate_extreme_ys_for_pt(barrier)
        led_blindness_matrix = self.led.blindness_matrix
        LED_matrix = self.led.matrix
        l0 = self.params.l0
        PT_intensity_barr = np.zeros_like(self.pt_mes.blindness_matrix)
        PT_matrix = self.pt_mes.matrix
        
        for i in range(PT_matrix.shape[0]):
            for j in range(PT_matrix.shape[1]):
                if self.pt_parameters.blind_shape[0] > i or self.pt_parameters.blind_shape[1] > j:
                    PT_intensity_barr[i, j] = calc_intensity_matrix_to_point(
                        PT_matrix[i, j],
                        LED_matrix,
                        led_map=led_blindness_matrix,
                        l_0=l0,
                        I_0=1,
                        y_max=extreme_ys[i],
                    )
        return PT_intensity_barr
    
    # def __visualize_matrix__(matrix: np.array):
    #     # Визуализируем матрицу 1
    #     for i in range(matrix.shape[0]):
    #         for j in range(matrix.shape[1]):
    #             x, y = matrix[i][j]
    #             scatter_list[0] = ax.scatter(
    #                 x,
    #                 y,
    #                 color=labels_colors[map_list["PT reference"]],
    #                 marker="s",
    #                 s=100,
    #                 edgecolors=labels_colors[map_list["PT reference"]] ,
    #                 alpha=0.8,
                    
    #                 # edgecolors="black",
    #                 label="PT reference",
    #             )

    def visualize(self, barrier=None, with_intensity:bool = True, save=True):
        pt_blindness = self.pt_ref.blindness_matrix
        PT_matrix_ref = self.pt_ref.matrix
        PT_matrix = self.pt_mes.matrix

        PT_matrix_intensity_ref = self.pt_ref.intensity_matrix
        PT_matrix_intensity = self.pt_mes.intensity_matrix
        LED_matrix = self.led.matrix
        led_blindness = self.led.blindness_matrix
        
        # Создаем график

        # csfont = {'fontname':'Times New Roman'}
        plt.rcParams["font.family"] = "Times New Roman"

        fig, ax = plt.subplots(figsize=(6, 6))

        labels_colors = [plt.cm.tab20(4), plt.cm.tab20(0), plt.cm.tab20(2) , plt.cm.tab20(0)]
        scatter_list = []
        scatter_labels = []
        edge_color = "#000001"
        text_color = plt.cm.tab20(14)
        labels = ["PT active cell", "PT passive cell", "LED active cell", "LED passive cell"]
        colors = ["red", "black", "red", "black", "orange", "indigo"]
        map_list = {
            "PT reference": 0,
            "PT reference passive cell": 1,
            "PT cell": 0,
            "PT passive cell": 1,
            "LED cell": 2,
            "LED passive cell": 3,
        }
        scatter_list = [0]*len(labels)
        zorder_blind = 10000
        zorder_visible = 1
        s_blind = 80
        s_visible = 100
        alpha_blind = 0.8
        alpha_visible = 0.1
        scatter_indx = None
        # Визуализируем матрицу 1
        for i in range(PT_matrix_ref.shape[0]):
            for j in range(PT_matrix_ref.shape[1]):
                x, y = PT_matrix_ref[i][j]
                alpha = 0
                label_color = 0
                zorder=0
                if pt_blindness[i, j] == 0:
                    scatter_indx = 1
                    s = s_blind
                    alpha = alpha_blind
                    zorder = zorder_blind
                    label_color = labels_colors[map_list["PT reference passive cell"]]
                else:
                    scatter_indx = 0
                    alpha = alpha_visible
                    s = s_visible
                    zorder= zorder_visible
                    label_color = labels_colors[map_list["PT reference"]]
                scatter_list[scatter_indx] = ax.scatter(
                    x,
                    y,
                    color=label_color,
                    marker="s",
                    s=s,
                    edgecolors=label_color ,
                    alpha=alpha,
                    zorder=zorder
                )
                
        # # Визуализируем матрицу 1
        # for i in range(PT_matrix_ref.shape[0]):
        #     for j in range(PT_matrix_ref.shape[1]):
        #         x, y = PT_matrix_ref[i][j]
        #         if pt_blindness[i, j] == 0:
        #             scatter_list[1] = ax.scatter(
        #                 x,
        #                 y,
        #                 color=labels_colors[map_list["PT reference passive cell"]],
        #                 marker="s",
        #                 s=100,
        #                 alpha=0.8,
                        
        #                 edgecolors=labels_colors[map_list["PT reference passive cell"]] ,

        #                 label="PT reference passive cell",
        #             )

        # Визуализируем матрицу 1
        for i in range(PT_matrix.shape[0]):
            for j in range(PT_matrix.shape[1]):
                x, y = PT_matrix[i][j]
                ax.scatter(
                    x,
                    y,
                    color=labels_colors[map_list["PT cell"]],
                    marker="s",
                    s=100,
                    edgecolors=labels_colors[map_list["PT cell"]] ,
                    alpha=0.8,
                    # edgecolors="black",
                    label="PT cell",
                )
        for i in range(PT_matrix.shape[0]):
            for j in range(PT_matrix.shape[1]):
                x, y = PT_matrix[i][j]
                if pt_blindness[i, j] == 0:
                    ax.scatter(
                        x, y, color=labels_colors[map_list["PT passive cell"]], 
                        marker="s", 
                        s=100, 
                        edgecolors=labels_colors[map_list["PT passive cell"]] ,
                        label="PT passive cell"
                    )

        # Визуализируем матрицу 2
        for i in range(LED_matrix.shape[0]):
            for j in range(LED_matrix.shape[1]):
                x, y = LED_matrix[i][j]
                scatter_list[2] = ax.scatter(
                    x,
                    y,
                    color=labels_colors[map_list["LED cell"]],
                    marker="s",
                    s=250,
                    alpha=0.8,
                
                    edgecolors=labels_colors[map_list["LED cell"]] ,
                    # alpha=0.8,
                    # edgecolors="black",
                    label="LED cell",
                )
        for i in range(LED_matrix.shape[0]):
            for j in range(LED_matrix.shape[1]):
                x, y = LED_matrix[i][j]
                if led_blindness[i, j] == 0:
                    scatter_list[3] = ax.scatter(
                        x,
                        y,
                        color=labels_colors[map_list["LED passive cell"]],
                        marker="s",
                        s=250,
                        # alpha=0.8,
                        edgecolors=labels_colors[map_list["LED passive cell"]] ,
                        
                        label="LED passive cell",
                    )



        x, y = np.random.rand(2, len(colors))

        legend1 = ax.legend(scatter_list, labels, loc='upper right')

        ax.annotate("PT Reference", xy=(PT_matrix_ref[0][int(self.pt_ref.param.n_columns/2)]), color=text_color, 
                    xytext=(-30,-16), textcoords="offset points", weight='bold',)

        ax.annotate("PT Measuring", xy=(PT_matrix[0][int(self.pt_mes.param.n_columns/2)]), color=text_color, 
                    xytext=(-30,-16), textcoords="offset points", weight='bold',)

        ax.annotate("LED", xy=(LED_matrix[0][int(self.led.param.n_columns/2)]), color=text_color, 
                    xytext=(-10,-20), textcoords="offset points", weight='bold',)

        if (isinstance(barrier, BarrierPosition)):
            X =  self.pt_parameters.X
            Y =  self.pt_parameters.Y
            xy = -X, barrier.y
            height = Y/2 - barrier.y
            width = X + 0.8

            ax.add_patch(Rectangle(xy, width=width, height=height, alpha=0.3))

        ax.add_artist(legend1)

        # ax.set_axis_off()
        ax.set_xlim((-0.8, 0.8))
        ax.set_ylim((-0.7, 0.7))
        if save:
            plot_name = '.png'
            if isinstance(barrier, BarrierPosition): plot_name = 'with_barrier' + plot_name
            if with_intensity: plot_name = 'intensity_'+plot_name
            
            plt.savefig("figs/"+plot_name)
            
        # Отображаем график
        plt.show()

    def visualize_intensity(self, barrier=None):
        plt.rcParams["font.family"] = "Times New Roman"
        pt_blindness = self.pt_ref.blindness_matrix
        PT_matrix_ref = self.pt_ref.matrix
        PT_matrix = self.pt_mes.matrix
        LED_blindness_matrix = self.led.blindness_matrix
        PT_matrix_intensity_ref = self.calculate_intensity(False)
        
        PT_intensity_barr = None
        if isinstance(barrier, BarrierPosition):
            PT_intensity_barr = self.calculate_intensity_with_barrier(barrier=barrier)
        else: PT_intensity_barr = self.calculate_intensity()
        
        LED_matrix = self.led.matrix
        led_blindness = self.led.blindness_matrix
        
        fig, ax = plt.subplots(figsize=(6, 5))

        labels_colors = [np.max(PT_intensity_barr), np.min(PT_intensity_barr), plt.cm.tab20(2) , plt.cm.tab20(0)]
        scatter_list = []
        scatter_labels = []
        edge_color = "#000001"
        text_color = plt.cm.tab20(14)
        labels = ["PT active cell", "PT passive cell", "LED active cell", "LED passive cell"]
        colors = ["red", "black", "red", "black", "orange", "indigo"]
        map_list = {
            "PT reference": 0,
            "PT reference passive cell": 1,
            "PT cell": 0,
            "PT passive cell": 1,
            "LED cell": 2,
            "LED passive cell": 3,
        }
        scatter_list = [0]*len(labels)

        # Визуализируем матрицу 1
        for i in range(PT_matrix_ref.shape[0]):
            for j in range(PT_matrix_ref.shape[1]):
                x, y = PT_matrix_ref[i][j]
                scatter_list[0] = ax.scatter(
                    x,
                    y,
                    c=labels_colors[map_list["PT reference"]],
                    marker="s",
                    s=25,
                    # edgecolors=labels_colors[map_list["PT reference"]] ,
                    alpha=0.8,
                    
                    # edgecolors="black",
                    label="PT reference",
                    vmin=np.min(PT_intensity_barr),
                    vmax=np.max(PT_intensity_barr),
                )
                
        for i in range(PT_matrix_ref.shape[0]):
            for j in range(PT_matrix_ref.shape[1]):
                x, y = PT_matrix_ref[i][j]
                if pt_blindness[i, j] == 0:
                    scatter_list[1] = ax.scatter(
                        x,
                        y,
                        c=labels_colors[map_list["PT reference passive cell"]],
                        marker="s",
                        s=25,
                        # edgecolors="black",
                        
                        alpha=1,
                        
                        # edgecolors=labels_colors[map_list["PT reference passive cell"]] ,

                        label="PT reference passive cell",
                        
                    )

        # Визуализируем матрицу 1
        for i in range(PT_matrix.shape[0]):
            for j in range(PT_matrix.shape[1]):
                x, y = PT_matrix[i][j]
                im = ax.scatter(
                    x,
                    y,
                    c=PT_intensity_barr[i, j],
                    marker="s",
                    s=25,
                    alpha=1 if PT_intensity_barr[i, j] == 0 else 0.8,
                    
                    # edgecolors='black',
                    vmin=np.min(PT_intensity_barr),
                    vmax=np.max(PT_intensity_barr),
                )
        for i in range(PT_matrix.shape[0]):
            for j in range(PT_matrix.shape[1]):
                x, y = PT_matrix[i][j]
                if pt_blindness[i, j] == 0:
                    im = ax.scatter(
                    x,
                    y,
                    c=0,
                    marker="s",
                    s=25,
                    # alpha=0.8 if PT_intensity_barr[i, j] + pt_blindness[i, j] == 0 else 1,
                    
                    # edgecolors='black',
                    vmin=np.min(PT_intensity_barr),
                    vmax=np.max(PT_intensity_barr),
                )
                
        # Визуализируем матрицу 2
        for i in range(LED_matrix.shape[0]):
            for j in range(LED_matrix.shape[1]):
                x, y = LED_matrix[i][j]
                scatter_list[2] = ax.scatter(
                    x,
                    y,
                    color=labels_colors[map_list["LED cell"]],
                    marker="s",
                    s=300,
                    # alpha=1,
                
                    edgecolors=labels_colors[map_list["LED cell"]] ,
                    # alpha=0.8,
                    # edgecolors="black",
                    label="LED cell",
                )
        for i in range(LED_matrix.shape[0]):
            for j in range(LED_matrix.shape[1]):
                x, y = LED_matrix[i][j]
                if LED_blindness_matrix[i, j] == 0:
                    scatter_list[3] = ax.scatter(
                        x,
                        y,
                        color=labels_colors[map_list["LED passive cell"]],
                        marker="s",
                        s=300,
                        # alpha=0.8,
                        edgecolors=labels_colors[map_list["LED passive cell"]] ,
                        
                        label="LED passive cell",
                    )


        ax.annotate("PT Reference", xy=(PT_matrix_ref[0][int(self.pt_ref.param.n_columns/2)]), color=text_color, 
                xytext=(-30,-16), textcoords="offset points", weight='bold',)

        ax.annotate("PT Measuring", xy=(PT_matrix[0][int(self.pt_mes.param.n_columns/2)]), color=text_color, 
                    xytext=(-30,-16), textcoords="offset points", weight='bold',)

        ax.annotate("LED", xy=(LED_matrix[0][int(self.led.param.n_columns/2)]), color=text_color, 
                    xytext=(-10,-20), textcoords="offset points", weight='bold',)

        ax.annotate("Photo Barrier", xy=(-self.pt_parameters.X, 0.4), color=text_color, 
                    xytext=(10, -20), textcoords="offset points", weight='bold',)


        if (isinstance(barrier, BarrierPosition)):
            X =  self.pt_parameters.X
            Y =  self.pt_parameters.Y
            xy = -X, barrier.y
            height = Y/2 - barrier.y
            width = X + 0.8

            ax.add_patch(Rectangle(xy, width=width, height=height, alpha=0.3))

        # cf = plt.contourf(PT_intensity_barr, levels=(0, 2.5, 5, 7.5, 10))
        # cb = fig.colorbar(im, ax=ax, label='$I$ normalized')
        # cb.set_ticklabels([0, 0.5, 1.00, round(np.min(PT_intensity_barr), 2),round(np.max(PT_intensity_barr), 2)])


        # fig.colorbar(im, ax=ax)

        # ax.add_artist(legend1)

        # ax.set_axis_off()
        ax.set_xlim((-0.6, 0.8))
        ax.set_ylim((-0.6, 0.6))

        # Отображаем график
        plt.show()

    
        
def divide_surface(surface_size: Tuple[int, int], n_rows, n_columns):
    '''
    The function divides the surface by n_row, n_columns and returns matrix of centers of the windows in form (x_i, y_i)
    '''
    R, C = surface_size
    r = R/n_columns
    c = C/n_rows

    # generator of coordinates of the windows centers
    r_array = np.arange(r/2, R, r)
    c_array = np.arange(c/2, C, c)

    x_matrix = np.repeat(r_array[np.newaxis, :], repeats=n_rows, axis=0)
    y_matrix = np.repeat(c_array[np.newaxis, :], repeats=n_columns, axis=0)

    return np.moveaxis(np.array([x_matrix, y_matrix.T]), 0, -1)

def make_blindness_matrix(surface_shape: Tuple[int, int], blind_shape, is_metal_plate_centered = True):
    '''
    Returns a true map matrix of size=blind_shape with 0 values. The matrix is located in the center of blindness matrix.
    '''

    n_rows, n_columns = surface_shape
    blindness_matrix = np.ones(shape=(n_rows, n_columns))
    k, l = blind_shape
    if is_metal_plate_centered:
        start_col = (n_columns - l) // 2
        start_row = (n_rows - k) // 2
        end_row = start_row + k
        end_col = start_col + l
    else:
        start_row = 0
        start_col = 0
        end_row = (n_rows - k) // 2
        end_col = (n_columns - l) // 2
    # Matrix shows the Intensity of n LED sectors
    blindness_matrix[start_row:end_row, start_col:end_col] = 0
    return blindness_matrix

def calc_intensity(emitter_xy: [float, float],  receiver_xy: [float, float], l_0: float=3.5, I_0: float = 1.0):
    x_l, y_l = emitter_xy
    x_p, y_p = receiver_xy

    R_2 = (x_l - x_p)**2 + (y_l - y_p)**2 + l_0**2
    h = y_p - y_l
    h = h*np.sign(h)
    theta = np.arccos(h/np.sqrt(R_2))
    # from my calculations
    # return I_0*(np.sin(theta)**2)/(4*np.pi*R_2)
    
    # From the paper
    return I_0*(np.sin(theta))/(4*np.pi*R_2/(2*l_0))

def calc_intensity_matrix_to_point(pt_xy: [float, float], led_matrix, led_map=None, l_0: float=3.5, I_0: float = 1.0, y_max: float = None):

    if led_map is None:
        led_map = np.ones(shape=(led_matrix.shape[0], led_matrix.shape[1]))
    
    intensity = 0
    for i in range(led_matrix.shape[0]):
        for j in range(led_matrix.shape[1]):
            x, y = led_matrix[i, j]
            if y_max is None or y < y_max:
                intensity+=calc_intensity(emitter_xy=led_matrix[i, j], receiver_xy=pt_xy) * led_map[i, j]
    return intensity

from scipy import integrate
import numpy as np

def calc_intensity_integration(emitter_x: float, emitter_y: float,  receiver_x: float, receiver_y: float, l_0: float=3.5, I_0: float = 1.0):
    x_l, y_l = emitter_x, emitter_y
    x_p, y_p = receiver_x, receiver_y

    R_2 = (x_l - x_p)**2 + (y_l - y_p)**2 + l_0**2
    h = y_p - y_l
    h = h*np.sign(h)
    theta = np.arccos(h/np.sqrt(R_2))
    # from my calculations
    # return I_0*(np.sin(theta)**2)/(4*np.pi*R_2)
    # From the paper
    return I_0*(np.sin(theta))/(4*np.pi*R_2/(2*l_0))

def calc_flux_on_area(source_matrix: np.ndarray, receiver_xy: tuple, dx:float, dy:float, led_shape: tuple):
    x, y = receiver_xy
    S1, S2 = led_shape
    lims = [[-S1/2, S1/2], [-S2/2, S2/2], [x-dx/2, x+dx/2], [y-dy/2, y+dy/2]]

    return integrate.nquad(calc_intensity_integration, ranges=lims)
