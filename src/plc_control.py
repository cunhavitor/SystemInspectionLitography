
import platform
import time

# Use gpiozero for Pi 5 compatibility
try:
    from gpiozero import DigitalOutputDevice, DigitalInputDevice, Device
    from gpiozero.pins.lgpio import LGPIOFactory
    # Force lgpio factory for Pi 5
    Device.pin_factory = LGPIOFactory()
    GPIO_AVAILABLE = True
except ImportError as e:
    GPIO_AVAILABLE = False
    print(f"WARNING: gpiozero/lgpio not found or failed to load: {e}. Running in MOCK GPIO mode.")

class MockOutputDevice:
    def __init__(self, pin, active_high=True, initial_value=False):
        self.pin = pin
        self._value = initial_value
    def on(self): self._value = True
    def off(self): self._value = False
    def toggle(self): self._value = not self._value
    @property
    def value(self): return self._value
    @property
    def is_active(self): return self._value

class MockInputDevice:
    def __init__(self, pin, pull_up=False):
        self.pin = pin
        self._value = False # Default state
    @property
    def value(self): return self._value
    @property
    def is_active(self): return self._value

class PLCManager:
    def __init__(self):
        # Outputs
        self.out_free = None  # GPIO 5
        self.out_valve = None # GPIO 6
        self.out_ready = None # GPIO 12
        self.out_fault = None # GPIO 13
        
        # Inputs
        self.in_machine_ready = None # GPIO 22
        self.in_trigger = None       # GPIO 23
        self.in_cycle = None         # GPIO 24
        self.in_reset = None         # GPIO 25
        
        self.setup_gpio()

    def setup_gpio(self):
        global GPIO_AVAILABLE
        
        # Determine if we are on a mock environment explicitly if needed,
        # otherwise trust the import result.
        
        outputs = [
            (5, 'out_free'),
            (6, 'out_valve'),
            (12, 'out_ready'),
            (13, 'out_fault')
        ]
        
        inputs = [
            (22, 'in_machine_ready'),
            (23, 'in_trigger'),
            (24, 'in_counter_cans'),
            (25, 'in_police_check')
        ]

        if GPIO_AVAILABLE:
            try:
                # Initialize Outputs
                for pin, name in outputs:
                    setattr(self, name, DigitalOutputDevice(pin, initial_value=False))
                    
                # Initialize Inputs
                # Check hardware specifics for pull-up/down needs. 
                # Assuming standard PLC 24V->3.3V optocoupler logic usually drives high or low actively.
                # Floating inputs might need pull selection. Defaulting to None (floating) or explicit if known.
                for pin, name in inputs:
                    # Using pull_up=None to use hardware default or external pull.
                    # Or set pull_up=False if we expect active High (3.3V) input.
                    # On Pi 5, gpiozero usually defaults nicely, but we can curb issues by being explicit if needed.
                    # For now just ensure we are creating the object correctly. 
                    # Use pull_up=None (floating) if inputs are driven actively 0/1.
                    # Try pulling UP. 
                    # If sensors are NPN (sink to ground), they need a Pull-Up.
                    # Idle = High (3.3V), Triggered = Low (GND).
                    # gpiozero with pull_up=True defaults active_state=False (Low),
                    # so is_active will be True when sensor triggers (grounds pin).
                    setattr(self, name, DigitalInputDevice(pin, pull_up=True))
                    
            except Exception as e:
                print(f"GPIO HW Init Failed: {e}. Falling back to MOCK.")
                GPIO_AVAILABLE = False
                self._init_mocks(outputs, inputs)
        else:
            self._init_mocks(outputs, inputs)

    def _init_mocks(self, outputs, inputs):
        for pin, name in outputs:
            setattr(self, name, MockOutputDevice(pin))
        for pin, name in inputs:
            setattr(self, name, MockInputDevice(pin))

    # --- Actions ---

    def set_valve_reject(self, state: bool):
        """GPIO 6: Valve to reject"""
        if self.out_valve:
            self.out_valve.on() if state else self.out_valve.off()

    def set_system_ready(self, state: bool):
        """GPIO 12: System Ready"""
        if self.out_ready:
            self.out_ready.on() if state else self.out_ready.off()

    def set_fault(self, state: bool):
        """GPIO 13: Fault / Alarm"""
        if self.out_fault:
            self.out_fault.on() if state else self.out_fault.off()

    # --- Status Reads ---

    def get_input_states(self):
        return {
            "Máquina Pronta (22)": self.in_machine_ready.is_active if self.in_machine_ready else False,
            "Trigger Captura (23)": self.in_trigger.is_active if self.in_trigger else False,
            "Contador Lata (24)": self.in_counter_cans.is_active if self.in_counter_cans else False,
            "Police Check (25)": self.in_police_check.is_active if self.in_police_check else False
        }
    
    def get_output_states(self):
        return {
            "Free (5)": self.out_free.value if self.out_free else False,
            "Válvula Rejeição (6)": self.out_valve.value if self.out_valve else False,
            "Sistema Pronto (12)": self.out_ready.value if self.out_ready else False,
            "Fault / Alarme (13)": self.out_fault.value if self.out_fault else False
        }
