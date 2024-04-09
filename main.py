from optic import *

def main():
    l0 = 3.5                      

    # Current height if the barrier lower edge
    y_barrier = 0.0
    # Depth of the barrier (along Z axis)
    barrier_depth = 0.01
    # Z position of the barrier (along the axis of direct light)
    z_barrier = l0 / 2
    parameters = OptoPairParameters()
    optopair = OptoPair(parameters)
    barrier = BarrierPosition(y = y_barrier, z =z_barrier, width=barrier_depth)
    # optopair.visualize(barrier=barrier)
    # optopair.visualize_intensity(barrier=barrier)
    # plt.savefig("figs/optopair_with_barrier.png")
    intensity = optopair.get_intensity(barrier)
    print(intensity)
if __name__=='__main__':
    main()