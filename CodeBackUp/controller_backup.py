class XppController_SD:
    """
    With this object, I define a lot of ways to access each motors.
    This certainly makes this object prone to error.
    However, I have little time to find a better solution.
    If you intend to use this future for your own work,
    you definitely need to rethink about the logic to make it compatible
    for your own applications

    """

    def __init__(self):
        # Step 1 Create all the optics and motors
        motors, optics = assemble_motors_and_optics(Ec=9.8)

        # Step 2 Create properties associate with each component
        self._motor_stacks = motors
        self._optics = optics

        self.t1 = motors['t1']
        self.t2 = motors['t2']
        self.t3 = motors['t3']
        self.t45 = motors['t45']
        self.t6 = motors['t6']
        self.g1 = motors['g1']
        self.g2 = motors['g2']
        self.tg_g = motors['tg g']
        self.m1 = motors['m1']
        self.m2a = motors['m2a']
        self.m2b = motors['m2b']
        self.si = motors['si']

        # For a quick and temporary solution.
        # I'll start with the aligned optics
        expSimu = rayTracingCalculation.get_miniSD_and_TG_trajectory()
        (t2x_coef,
         t3x_coef,
         t4x_coef,
         t5x_coef,
         probe1z_coef,
         probe1alpha_coef,
         probe2z_coef,
         pump1z_coef,
         pump1alpha_coef,
         pump2z_coef,
         pump2alpha_coef) = rayTracingCalculation.get_trajectory_dependence_on_various_parameters()

        # controller.cc1 = optics['cc1']
        # controller.cc2 = optics['cc2']
        # controller.vcc1 = optics['vcc1']
        # controller.vcc2 = optics['vcc2']
        # controller.vcc3 = optics['vcc3']
        # controller.vcc4 = optics['vcc4']

        self.cc1 = expSimu['devices']['cc'][0]
        self.cc2 = expSimu['devices']['cc'][1]
        self.vcc1 = expSimu['devices']['vcc'][0]
        self.vcc2 = expSimu['devices']['vcc'][1]
        self.vcc3 = expSimu['devices']['vcc'][2]
        self.vcc4 = expSimu['devices']['vcc'][3]

        # This a temporary entry which contains the aligned optics from the old method
        self.expSimu = expSimu

        # Add the shutter
        self.cc_shutter = True
        self.vcc_shutter = True

        # Step 3 Move the devices to their rough position
        self.t1.insatll()

        # Step 4 Rotate the crystals such that they are at the ideal path

        # Step 5 Add diodes

        # Step 6 Add cameras
        self.pixel_num_x = 2048
        self.pixel_num_y = 2048

        self.pixel_pos_x = np.linspace(- 1024 * 6.5 / 3, 1024 * 6.5 / 3, 2048) + expSimu['cc']['trajectory'][-1, 0]
        self.pixel_pos_y = np.linspace(- 1024 * 6.5 / 3, 1024 * 6.5 / 3, 2048) + expSimu['cc']['trajectory'][-1, 1]

    def plot_motors(self):
        pass

    def get_diode(self):
        pass

    def get_camera(self):
        pass

    def show_cc(self):
        self.cc_shutter = True
        self.vcc_shutter = False

    def show_vcc(self):
        self.vcc_shutter = True
        self.cc_shutter = False

    def show_both(self):
        self.vcc_shutter = True
        self.cc_shutter = True

    def show_neither(self):
        self.vcc_shutter = False
        self.cc_shutter = False
