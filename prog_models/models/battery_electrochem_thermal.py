from prog_models.models.battery_electrochem import BatteryElectroChem

class BatteryElectroChemThermal(BatteryElectroChem):
    """
    Prognostics model for a battery, represented by an electrochemical equations with added thermal model 

    See:
        BatteryElectroChem

    Args:
        BatteryElectroChem ([type]): [description]
    """
    def dx(self, x, u):
        dx = super().dx(x, u)
        # Thermal affects (from TBD)
        voltage_eta = x['Vo'] + x['Vsn'] + x['Vsp'] # (Vep - Ven) - V;
        Tb0 = 293.15 # K
        mC = 37.04 # kg/m2/(K-s^2)
        tau = 100

        dx['tb'] = (voltage_eta*u['i']/mC) + ((Tb0 - x['tb'])/tau) # Newman
        return dx
