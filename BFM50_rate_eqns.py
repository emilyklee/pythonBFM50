from __future__ import division, print_function
import numpy
import json
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from Functions.seasonal_cycling_functions import get_wind, get_salinity, get_sunlight, get_temperature, calculate_density
from Functions.phyto import phyto_eqns
from Functions.bacteria import bacteria_eqns
from Functions.predation import get_mesozoo_predation_terms, get_microzoo_predation_terms
from Functions.micro import microzoo_eqns
from Functions.meso import mesozoo_eqns
from Functions.pel_chem import pel_chem_eqns
from Functions.oxygen import calculate_oxygen_reaeration
from Functions.co2_flux_functions import calculate_co2_flux
from Functions.other_functions import insw_vector, get_concentration_ratio


# Names of species in the system
species_names = ['O2o', 'N1p', 'N3n', 'N4n', 'O4n', 'N5s', 'N6r', 'B1c', 'B1n', 'B1p', 
                 'P1c', 'P1n', 'P1p', 'P1l', 'P1s', 'P2c', 'P2n', 'P2p', 'P2l',
                 'P3c', 'P3n', 'P3p', 'P3l', 'P4c', 'P4n', 'P4p', 'P4l',
                 'Z3c', 'Z3n', 'Z3p', 'Z4c', 'Z4n', 'Z4p', 'Z5c', 'Z5n', 'Z5p',
                 'Z6c', 'Z6n', 'Z6p', 'R1c', 'R1n', 'R1p', 'R2c', 'R3c', 'R6c', 
                 'R6n', 'R6p', 'R6s', 'O3c', 'O3h']


def bfm50_rate_eqns(time, conc, seasonal_cycle=True):
    """ Calculates the change in concentration for the 50 state variables
        NOTE: iron dynamics are not included, this is to compare to the standalone pelagic system
    """



    #--------------------------------------------------------------------------
    # import parameters from json file
    with open("bfm50_parameters.json", "r") as read_parameters:
        parameters = json.load(read_parameters)
    
    constant_parameters = parameters["constants"]
    environmental_parameters = parameters["environmental_parameters"]
    bacteria_parameters = parameters["bacteria_parameters"]
    phyto1_prameters = parameters["phyto1_parameters"]
    phyto2_prameters = parameters["phyto2_parameters"]
    phyto3_prameters = parameters["phyto3_parameters"]
    phyto4_prameters = parameters["phyto4_parameters"]
    mesozoo3_parameters = parameters["mesozoo3_parameters"]
    mesozoo4_parameters = parameters["mesozoo4_parameters"]
    microzoo5_parameters = parameters["microzoo5_parameters"]
    microzoo6_parameters = parameters["microzoo6_parameters"]
    zoo_availability_parameters = parameters["zoo_availability_parameters"]
    pel_chem_parameters = parameters["pel_chem_parameters"]
    co2_flux_parameters = parameters["co2_flux_parameters"]
    oxygen_reaeration_parameters = parameters["oxygen_reaeration_parameters"]
    
    #--------------------------------------------------------------------------
    # Seasonal wind, temp, salinity, and radiation values
    if seasonal_cycle:

        t = time

        # Wind
        w_win = environmental_parameters["w_win"]                               # Winter wind speed
        w_sum = environmental_parameters["w_sum"]                               # Summer wind speed
        wind = get_wind(t,w_win,w_sum)                                          # Yearly wind cylce

        # Temperature
        t_win = environmental_parameters["t_win"]                               # Winter temp value
        t_sum = environmental_parameters["t_sum"]                               # Summer temp value
        tde = environmental_parameters["tde"]                                   # Sinusoidal temperature daily excursion degC
        temper = get_temperature(t,t_win,t_sum, tde)                            # Yearly temp cycle

        # Salinity
        s_win = environmental_parameters["s_win"]                               # Winter salinity value
        s_sum = environmental_parameters["s_sum"]                               # Summer salinity value
        salt = get_salinity(t,s_win,s_sum)                                      # Yearly salinity cycle

        # Short wave irradiance flux (W/m^2)
        qs_win = environmental_parameters["qs_win"]                             # Winter irradiance value
        qs_sum = environmental_parameters["qs_sum"]                             # Summer irradiance value
        qs = get_sunlight(t,qs_win,qs_sum)                                      # Yearly irradiance cycle

    else:
        wind = environmental_parameters["w_win"]
        temper = environmental_parameters["t_win"]
        salt = environmental_parameters["s_win"]
        qs = environmental_parameters["qs_win"]

    #--------------------------------------------------------------------------
    # State variables
    o2o = conc[0]              # Dissolved oxygen (mg O_2 m^-3)
    n1p = conc[1]              # Phosphate (mmol P m^-3)
    n3n = conc[2]              # Nitrate (mmol N m^-3)
    n4n = conc[3]              # Ammonium (mmol N m^-3)
    o4n = conc[4]              # Nitrogen sink (mmol N m^-3)
    n5s = conc[5]              # Silicate (mmol Si m^-3)
    n6r = conc[6]              # Reduction equivalents (mmol S m^-3)
    b1c = conc[7]              # Pelagic bacteria carbon (mg C m^-3)
    b1n = conc[8]              # Pelagic bacteria nitrogen (mmol N m^-3)
    b1p = conc[9]              # Pelagic bacteria phosphate (mmol P m^-3)
    p1c = conc[10]             # Diatoms carbon (mg C m^-3)
    p1n = conc[11]             # Diatoms nitrogen (mmol N m^-3)
    p1p = conc[12]             # Diatoms phosphate (mmol P m^-3)
    p1l = conc[13]             # Diatoms chlorophyll (mg Chl-a m^-3)
    p1s = conc[14]             # Diatoms silicate (mmol Si m^-3) 
    p2c = conc[15]             # NanoFlagellates carbon (mg C m^-3)
    p2n = conc[16]             # NanoFlagellates nitrogen (mmol N m^-3)
    p2p = conc[17]             # NanoFlagellates phosphate (mmol P m^-3)
    p2l = conc[18]             # NanoFlagellates chlorophyll (mg Chl-a m^-3)
    p3c = conc[19]             # Picophytoplankton carbon (mg C m^-3)
    p3n = conc[20]             # Picophytoplankton nitrogen (mmol N m^-3)
    p3p = conc[21]             # Picophytoplankton phosphate (mmol P m^-3)
    p3l = conc[22]             # Picophytoplankton chlorophyll (mg Chl-a m^-3)
    p4c = conc[23]             # Large phytoplankton carbon (mg C m^-3)
    p4n = conc[24]             # Large phytoplankton nitrogen (mmol N m^-3)
    p4p = conc[25]             # Large phytoplankton phosphate (mmol P m^-3) 
    p4l = conc[26]             # Large phytoplankton chlorophyll (mg Chl-a m^-3)
    z3c = conc[27]             # Carnivorous mesozooplankton carbon (mg C m^-3)
    z3n = conc[28]             # Carnivorous mesozooplankton nitrogen (mmol N m^-3)
    z3p = conc[29]             # Carnivorous mesozooplankton phosphate (mmol P m^-3)
    z4c = conc[30]             # Omnivorous mesozooplankton carbon (mg C m^-3)
    z4n = conc[31]             # Omnivorous mesozooplankton nitrogen (mmol N m^-3)
    z4p = conc[32]             # Omnivorous mesozooplankton phosphate (mmol P m^-3)
    z5c = conc[33]             # Microzooplankton carbon (mg C m^-3)
    z5n = conc[34]             # Microzooplankton nitrogen (mmol N m^-3)
    z5p = conc[35]             # Microzooplankton phosphate (mmol P m^-3)
    z6c = conc[36]             # Heterotrophic flagellates carbon (mg C m^-3)
    z6n = conc[37]             # Heterotrophic flagellates nitrogen (mmol N m^-3)
    z6p = conc[38]             # Heterotrophic flagellates phosphate (mmol P m^-3)
    r1c = conc[39]             # Labile dissolved organic carbon (mg C m^-3)
    r1n = conc[40]             # Labile dissolved organic nitrogen (mmol N m^-3)
    r1p = conc[41]             # Labile dissolved organic phosphate (mmol P m^-3)
    r2c = conc[42]             # Semi-labile dissolved organic carbon (mg C m^-3)
    r3c = conc[43]             # Semi-refractory Dissolved Organic Carbon (mg C m^-3)
    r6c = conc[44]             # Particulate organic carbon (mg C m^-3)
    r6n = conc[45]             # Particulate organic nitrogen (mmol N m^-3)
    r6p = conc[46]             # Particulate organic phosphate (mmol P m^-3)
    r6s = conc[47]             # Particulate organic silicate (mmol Si m^-3)
    o3c = conc[48]             # Dissolved inorganic carbon(mg C m^-3)
    o3h = conc[49]             # Total alkalinity (mmol Eq m^-3)

    #--------------------------------------------------------------------------
    # concentration ratios
    p1n_p1c = get_concentration_ratio(p1n, p1c, constant_parameters["p_small"])
    p1p_p1c = get_concentration_ratio(p1p, p1c, constant_parameters["p_small"])
    p1l_p1c = get_concentration_ratio(p1l, p1c, constant_parameters["p_small"])
    p1s_p1c = get_concentration_ratio(p1s, p1c, constant_parameters["p_small"])
    p2n_p2c = get_concentration_ratio(p2n, p2c, constant_parameters["p_small"])
    p2p_p2c = get_concentration_ratio(p2p, p2c, constant_parameters["p_small"])
    p2l_p2c = get_concentration_ratio(p2l, p2c, constant_parameters["p_small"])
    p3n_p3c = get_concentration_ratio(p3n, p3c, constant_parameters["p_small"])
    p3p_p3c = get_concentration_ratio(p3p, p3c, constant_parameters["p_small"])
    p3l_p3c = get_concentration_ratio(p3l, p3c, constant_parameters["p_small"])
    p4n_p4c = get_concentration_ratio(p4n, p4c, constant_parameters["p_small"])
    p4p_p4c = get_concentration_ratio(p4p, p4c, constant_parameters["p_small"])
    p4l_p4c = get_concentration_ratio(p4l, p4c, constant_parameters["p_small"])
    bp_bc = get_concentration_ratio(b1p, b1c, constant_parameters["p_small"])
    bn_bc = get_concentration_ratio(b1n, b1c, constant_parameters["p_small"])
    z3n_z3c = get_concentration_ratio(z3n, z3c, constant_parameters["p_small"])
    z3p_z3c = get_concentration_ratio(z3p, z3c, constant_parameters["p_small"])
    z4n_z4c = get_concentration_ratio(z4n, z4c, constant_parameters["p_small"])
    z4p_z4c = get_concentration_ratio(z4p, z4c, constant_parameters["p_small"])
    z5n_z5c = get_concentration_ratio(z5n, z5c, constant_parameters["p_small"])
    z5p_z5c = get_concentration_ratio(z5p, z5c, constant_parameters["p_small"])
    z6n_z6c = get_concentration_ratio(z6n, z6c, constant_parameters["p_small"])
    z6p_z6c = get_concentration_ratio(z6p, z6c, constant_parameters["p_small"])
    r1p_r1c = get_concentration_ratio(r1p, r1c, constant_parameters["p_small"])
    r6p_r6c = get_concentration_ratio(r6p, r6c, constant_parameters["p_small"])
    r1n_r1c = get_concentration_ratio(r1n, r1c, constant_parameters["p_small"])
    r6n_r6c = get_concentration_ratio(r6n, r6c, constant_parameters["p_small"])

    #--------------------------------------------------------------------------
    #---------------------- Phytoplankton Equations ---------------------------
    #--------------------------------------------------------------------------
    # P1: Diatoms terms
    (dP1cdt_gpp_o3c, dP1cdt_rsp_o3c, dP1cdt_lys_r1c, dP1cdt_lys_r6c, dP1cdt_exu_r2c, dP1ndt_upt_n3n, dP1ndt_upt_n4n, 
     extra_n1, dP1ndt_lys_r1n, dP1ndt_lys_r6n, dP1pdt_upt_n1p, dP1pdt_upt_r1p, dP1pdt_lys_r1p, dP1pdt_lys_r6p, 
     dP1ldt_syn, dP1sdt_upt_n5s, dP1sdt_lys_r6s) = phyto_eqns(conc, phyto1_prameters, environmental_parameters, constant_parameters, 1, p1c, p1n, p1p, p1l, qs, temper, time)

    # P2: Flagellates terms
    (dP2cdt_gpp_o3c, dP2cdt_rsp_o3c, dP2cdt_lys_r1c, dP2cdt_lys_r6c, dP2cdt_exu_r2c, dP2ndt_upt_n3n, dP2ndt_upt_n4n, 
     extra_n2, dP2ndt_lys_r1n, dP2ndt_lys_r6n, dP2pdt_upt_n1p, dP2pdt_upt_r1p, dP2pdt_lys_r1p, dP2pdt_lys_r6p, 
     dP2ldt_syn, dP2sdt_upt_n5s, dP2sdt_lys_r6s) = phyto_eqns(conc, phyto2_prameters, environmental_parameters, constant_parameters, 2, p2c, p2n, p2p, p2l, qs, temper, time)

    # P3: PicoPhytoplankton terms
    (dP3cdt_gpp_o3c, dP3cdt_rsp_o3c, dP3cdt_lys_r1c, dP3cdt_lys_r6c, dP3cdt_exu_r2c, dP3ndt_upt_n3n, dP3ndt_upt_n4n, 
     extra_n3, dP3ndt_lys_r1n, dP3ndt_lys_r6n, dP3pdt_upt_n1p, dP3pdt_upt_r1p, dP3pdt_lys_r1p, dP3pdt_lys_r6p, 
     dP3ldt_syn, dP3sdt_upt_n5s, dP3sdt_lys_r6s) = phyto_eqns(conc, phyto3_prameters, environmental_parameters, constant_parameters, 3, p3c, p3n, p3p, p3l, qs, temper, time)

    # P4: Large Phytoplankton terms
    (dP4cdt_gpp_o3c, dP4cdt_rsp_o3c, dP4cdt_lys_r1c, dP4cdt_lys_r6c, dP4cdt_exu_r2c, dP4ndt_upt_n3n, dP4ndt_upt_n4n, 
     extra_n4, dP4ndt_lys_r1n, dP4ndt_lys_r6n, dP4pdt_upt_n1p, dP4pdt_upt_r1p, dP4pdt_lys_r1p, dP4pdt_lys_r6p, 
     dP4ldt_syn, dP4sdt_upt_n5s, dP4sdt_lys_r6s) = phyto_eqns(conc, phyto4_prameters, environmental_parameters, constant_parameters, 4, p4c, p4n, p4p, p4l, qs, temper, time)

    #--------------------------------------------------------------------------
    #------------------------- Bacteria Equations -----------------------------
    #--------------------------------------------------------------------------
    (dBcdt_lys_r1c, dBcdt_lys_r1n, dBcdt_lys_r1p, dBcdt_lys_r6c, dBcdt_lys_r6n, dBcdt_lys_r6p, 
     dBcdt_upt_r1c, dBcdt_upt_r6c, dBpdt_upt_rel_n1p, dBndt_upt_rel_n4n, dBcdt_upt_r2c, dBcdt_upt_r3c, 
     dBcdt_rel_r2c, dBcdt_rel_r3c, dBcdt_rsp_o3c, flPTN6r, f_B_O, f_B_n, f_B_p) = bacteria_eqns(conc, bacteria_parameters, constant_parameters, environmental_parameters, temper)

    #--------------------------------------------------------------------------
    #----------------------- Zooplankton Equations ----------------------------
    #--------------------------------------------------------------------------
    
    # Mesozooplankton predation terms
    dZ3cdt_prd, dZ4cdt_prd, ic3, in3, ip3, ic4, in4, ip4 = get_mesozoo_predation_terms(conc, mesozoo3_parameters, mesozoo4_parameters, zoo_availability_parameters, environmental_parameters, constant_parameters, temper)

    # Microzooplankton predation terms
    dZ5cdt_prd, dZ6cdt_prd, ic5, in5, ip5, ic6, in6, ip6 = get_microzoo_predation_terms(conc, microzoo5_parameters, microzoo6_parameters, zoo_availability_parameters, environmental_parameters, constant_parameters, temper)

    # Z3: Carnivorous Mesozooplankton terms
    (dZ3cdt_rel_r1c, dZ3cdt_rel_r6c, dZ3cdt_rsp_o3c, dZ3ndt_rel_r1n, dZ3ndt_rel_r6n, dZ3pdt_rel_r1p, dZ3pdt_rel_r6p, 
     dZ3pdt_rel_n1p, dZ3ndt_rel_n4n) = mesozoo_eqns(conc, mesozoo3_parameters, constant_parameters, environmental_parameters, z3c, z3n, z3p, ic3, in3, ip3, temper)

    # Z4: Omnivorous Mesozooplankton terms
    (dZ4cdt_rel_r1c, dZ4cdt_rel_r6c, dZ4cdt_rsp_o3c, dZ4ndt_rel_r1n, dZ4ndt_rel_r6n, dZ4pdt_rel_r1p, dZ4pdt_rel_r6p, 
     dZ4pdt_rel_n1p, dZ4ndt_rel_n4n) = mesozoo_eqns(conc, mesozoo4_parameters, constant_parameters, environmental_parameters, z4c, z4n, z4p, ic4, in4, ip4, temper)

    # Z5: Microzooplankton terms
    (dZ5cdt_rel_r1c, dZ5cdt_rel_r6c, dZ5cdt_rsp_o3c, dZ5ndt_rel_r1n, dZ5ndt_rel_r6n, dZ5pdt_rel_r1p, dZ5pdt_rel_r6p, 
     dZ5pdt_rel_n1p, dZ5ndt_rel_n4n) = microzoo_eqns(conc, microzoo5_parameters, constant_parameters, environmental_parameters, z5c, z5n, z5p, ic5, in5, ip5, temper)

    # Z6: Heterotrophic Nanoflagellates terms
    (dZ6cdt_rel_r1c, dZ6cdt_rel_r6c, dZ6cdt_rsp_o3c, dZ6ndt_rel_r1n, dZ6ndt_rel_r6n, dZ6pdt_rel_r1p, dZ6pdt_rel_r6p,  
     dZ6pdt_rel_n1p, dZ6ndt_rel_n4n) = microzoo_eqns(conc, microzoo6_parameters, constant_parameters, environmental_parameters, z6c, z6n, z6p, ic6, in6, ip6, temper)

    #--------------------------------------------------------------------------
    #------------------------ Non-living components ---------------------------
    #--------------------------------------------------------------------------    
    (dn4ndt_nit_n3n, dn3ndt_denit, dn6rdt_reox, dr6sdt_rmn_n5s) = pel_chem_eqns(pel_chem_parameters, environmental_parameters, constant_parameters, temper, conc, flPTN6r)

    #---------------------- Oxygen airation by wind ---------------------------
    
    dOdt_wind = calculate_oxygen_reaeration(oxygen_reaeration_parameters, environmental_parameters, constant_parameters, conc, temper, salt, wind)

    #------------------------------- CO_2 Flux --------------------------------
    do3cdt_air_sea_flux = calculate_co2_flux(co2_flux_parameters, environmental_parameters, constant_parameters, conc, temper, wind, salt)

    #--------------------------------------------------------------------------
    #----------------------------- Rate Equations -----------------------------
    #--------------------------------------------------------------------------
    
    # Dissolved oxygen [mmol O_2 m^-3 s^-1]
    do2o_dt = (constant_parameters["omega_c"]*((dP1cdt_gpp_o3c - dP1cdt_rsp_o3c) + (dP2cdt_gpp_o3c - dP2cdt_rsp_o3c) + (dP3cdt_gpp_o3c - dP3cdt_rsp_o3c) + (dP4cdt_gpp_o3c - dP4cdt_rsp_o3c)) - 
               constant_parameters["omega_c"]*f_B_O*dBcdt_rsp_o3c - 
               constant_parameters["omega_c"]*(dZ3cdt_rsp_o3c + dZ4cdt_rsp_o3c + dZ5cdt_rsp_o3c + dZ6cdt_rsp_o3c) - 
               constant_parameters["omega_n"]*dn4ndt_nit_n3n -
               (1.0/constant_parameters["omega_r"])*dn6rdt_reox) + dOdt_wind
    
    # Dissolved inorganic nutrient equations
    dn1p_dt = - (dP1pdt_upt_n1p + dP2pdt_upt_n1p + dP3pdt_upt_n1p + dP4pdt_upt_n1p) + (dBpdt_upt_rel_n1p*insw_vector(dBpdt_upt_rel_n1p)) - ((-1)*f_B_p*dBpdt_upt_rel_n1p*insw_vector(-dBpdt_upt_rel_n1p)) + (dZ3pdt_rel_n1p + dZ4pdt_rel_n1p + dZ5pdt_rel_n1p + dZ6pdt_rel_n1p)
    dn3n_dt = - (dP1ndt_upt_n3n + dP2ndt_upt_n3n + dP3ndt_upt_n3n + dP4ndt_upt_n3n) + dn4ndt_nit_n3n - dn3ndt_denit
    dn4n_dt = - (dP1ndt_upt_n4n + dP2ndt_upt_n4n + dP3ndt_upt_n4n + dP4ndt_upt_n4n) + (dBndt_upt_rel_n4n*insw_vector(dBndt_upt_rel_n4n)) - ((-1)*f_B_n*dBndt_upt_rel_n4n*insw_vector(-dBndt_upt_rel_n4n)) + (dZ3ndt_rel_n4n + dZ4ndt_rel_n4n + dZ5ndt_rel_n4n + dZ6ndt_rel_n4n) - dn4ndt_nit_n3n
    do4n_dt = dn3ndt_denit
    dn5s_dt = - dP1sdt_upt_n5s + dr6sdt_rmn_n5s

    # Reduction equivalents
    dn6r_dt = constant_parameters["omega_r"]*constant_parameters["omega_c"]*(1.0 - f_B_O)*dBcdt_rsp_o3c - constant_parameters["omega_r"]*constant_parameters["omega_dn"]*dn3ndt_denit*insw_vector(-(o2o - n6r)/constant_parameters["omega_r"]) - dn6rdt_reox

    # Bacterioplankton
    db1c_dt = dBcdt_upt_r1c + dBcdt_upt_r6c - dBcdt_rsp_o3c - dBcdt_lys_r1c - dBcdt_lys_r6c - (dZ5cdt_prd["b1"] + dZ6cdt_prd["b1"])
    db1n_dt = - dBcdt_lys_r1n - dBcdt_lys_r6n + (r1n_r1c*dBcdt_upt_r1c) + (r6n_r6c*dBcdt_upt_r6c) - (dBndt_upt_rel_n4n*insw_vector(dBndt_upt_rel_n4n)) + ((-1)*f_B_n*dBndt_upt_rel_n4n*insw_vector(-dBndt_upt_rel_n4n)) - (bn_bc*(dZ5cdt_prd["b1"] + dZ6cdt_prd["b1"]))
    db1p_dt = (r1p_r1c*dBcdt_upt_r1c) + (r6p_r6c*dBcdt_upt_r6c) - (dBpdt_upt_rel_n1p*insw_vector(dBpdt_upt_rel_n1p)) + ((-1)*f_B_p*dBpdt_upt_rel_n1p*insw_vector(-dBpdt_upt_rel_n1p)) - dBcdt_lys_r1p - dBcdt_lys_r6p - (bp_bc*(dZ5cdt_prd["b1"] + dZ6cdt_prd["b1"]))

    # Phytoplankton
    dp1c_dt = dP1cdt_gpp_o3c - dP1cdt_exu_r2c - dP1cdt_rsp_o3c - dP1cdt_lys_r1c - dP1cdt_lys_r6c - dZ3cdt_prd["p1"] - dZ4cdt_prd["p1"] - dZ5cdt_prd["p1"] - dZ6cdt_prd["p1"]
    dp1n_dt = dP1ndt_upt_n3n + dP1ndt_upt_n4n - extra_n1 - dP1ndt_lys_r1n - dP1ndt_lys_r6n - (p1n_p1c*(dZ3cdt_prd["p1"] + dZ4cdt_prd["p1"] + dZ5cdt_prd["p1"] + dZ6cdt_prd["p1"]))
    dp1p_dt = dP1pdt_upt_n1p - dP1pdt_upt_r1p - dP1pdt_lys_r1p - dP1pdt_lys_r6p - (p1p_p1c*(dZ3cdt_prd["p1"] + dZ4cdt_prd["p1"] + dZ5cdt_prd["p1"] + dZ6cdt_prd["p1"]))
    dp1l_dt = dP1ldt_syn - (p1l_p1c*(dZ3cdt_prd["p1"] + dZ4cdt_prd["p1"] + dZ5cdt_prd["p1"] + dZ6cdt_prd["p1"]))
    dp1s_dt = dP1sdt_upt_n5s - dP1sdt_lys_r6s - (p1s_p1c*(dZ3cdt_prd["p1"] + dZ4cdt_prd["p1"] + dZ5cdt_prd["p1"] + dZ6cdt_prd["p1"]))
    
    dp2c_dt = dP2cdt_gpp_o3c - dP2cdt_exu_r2c - dP2cdt_rsp_o3c - dP2cdt_lys_r1c - dP2cdt_lys_r6c - dZ3cdt_prd["p2"] - dZ4cdt_prd["p2"] - dZ5cdt_prd["p2"] - dZ6cdt_prd["p2"]
    dp2n_dt = dP2ndt_upt_n3n + dP2ndt_upt_n4n - extra_n2 - dP2ndt_lys_r1n - dP2ndt_lys_r6n - (p2n_p2c*(dZ3cdt_prd["p2"] + dZ4cdt_prd["p2"] + dZ5cdt_prd["p2"] + dZ6cdt_prd["p2"]))
    dp2p_dt = dP2pdt_upt_n1p - dP2pdt_upt_r1p - dP2pdt_lys_r1p - dP2pdt_lys_r6p - (p2p_p2c*(dZ3cdt_prd["p2"] + dZ4cdt_prd["p2"] + dZ5cdt_prd["p2"] + dZ6cdt_prd["p2"]))
    dp2l_dt = dP2ldt_syn - (p2l_p2c*(dZ3cdt_prd["p2"] + dZ4cdt_prd["p2"] + dZ5cdt_prd["p2"] + dZ6cdt_prd["p2"]))
    
    dp3c_dt = dP3cdt_gpp_o3c - dP3cdt_exu_r2c - dP3cdt_rsp_o3c - dP3cdt_lys_r1c - dP3cdt_lys_r6c - dZ3cdt_prd["p3"] - dZ4cdt_prd["p3"] - dZ5cdt_prd["p3"] - dZ6cdt_prd["p3"]
    dp3n_dt = dP3ndt_upt_n3n + dP3ndt_upt_n4n - extra_n3 - dP3ndt_lys_r1n - dP3ndt_lys_r6n - (p3n_p3c*(dZ3cdt_prd["p3"] + dZ4cdt_prd["p3"] + dZ5cdt_prd["p3"] + dZ6cdt_prd["p3"]))
    dp3p_dt = dP3pdt_upt_n1p - dP3pdt_upt_r1p - dP3pdt_lys_r1p - dP3pdt_lys_r6p - (p3p_p3c*(dZ3cdt_prd["p3"] + dZ4cdt_prd["p3"] + dZ5cdt_prd["p3"] + dZ6cdt_prd["p3"]))
    dp3l_dt = dP3ldt_syn - (p3l_p3c*(dZ3cdt_prd["p3"] + dZ4cdt_prd["p3"] + dZ5cdt_prd["p3"] + dZ6cdt_prd["p3"]))
    
    dp4c_dt = dP4cdt_gpp_o3c - dP4cdt_exu_r2c - dP4cdt_rsp_o3c - dP4cdt_lys_r1c - dP4cdt_lys_r6c - dZ3cdt_prd["p4"] - dZ4cdt_prd["p4"] - dZ5cdt_prd["p4"] - dZ6cdt_prd["p4"]
    dp4n_dt = dP4ndt_upt_n3n + dP4ndt_upt_n4n - extra_n4 - dP4ndt_lys_r1n - dP4ndt_lys_r6n - (p4n_p4c*(dZ3cdt_prd["p4"] + dZ4cdt_prd["p4"] + dZ5cdt_prd["p4"] + dZ6cdt_prd["p4"]))
    dp4p_dt = dP4pdt_upt_n1p - dP4pdt_upt_r1p - dP4pdt_lys_r1p - dP4pdt_lys_r6p - (p4p_p4c*(dZ3cdt_prd["p4"] + dZ4cdt_prd["p4"] + dZ5cdt_prd["p4"] + dZ6cdt_prd["p4"]))
    dp4l_dt = dP4ldt_syn - (p4l_p4c*(dZ3cdt_prd["p4"] + dZ4cdt_prd["p4"] + dZ5cdt_prd["p4"] + dZ6cdt_prd["p4"]))

    # mesozooplankton
    dz3c_dt = dZ3cdt_prd["p1"] + dZ3cdt_prd["p2"] + dZ3cdt_prd["p3"] + dZ3cdt_prd["p4"] + dZ3cdt_prd["z4"] + dZ3cdt_prd["z5"] + dZ3cdt_prd["z6"] - dZ4cdt_prd["z3"] - dZ3cdt_rel_r1c - dZ3cdt_rel_r6c - dZ3cdt_rsp_o3c
    dz3n_dt = p1n_p1c*dZ3cdt_prd["p1"] + p2n_p2c*dZ3cdt_prd["p2"] + p3n_p3c*dZ3cdt_prd["p3"] + p4n_p4c*dZ3cdt_prd["p4"] + z4n_z4c*dZ3cdt_prd["z4"] + z5n_z5c*dZ3cdt_prd["z5"] + z6n_z6c*dZ3cdt_prd["z6"] - z3n_z3c*dZ4cdt_prd["z3"] - dZ3ndt_rel_r1n - dZ3ndt_rel_r6n - dZ3ndt_rel_n4n
    dz3p_dt = p1p_p1c*dZ3cdt_prd["p1"] + p2p_p2c*dZ3cdt_prd["p2"] + p3p_p3c*dZ3cdt_prd["p3"] + p4p_p4c*dZ3cdt_prd["p4"] + z4p_z4c*dZ3cdt_prd["z4"] + z5p_z5c*dZ3cdt_prd["z5"] + z6p_z6c*dZ3cdt_prd["z6"] - z3p_z3c*dZ4cdt_prd["z3"] - dZ3pdt_rel_r1p - dZ3pdt_rel_r6p - dZ3pdt_rel_n1p
    
    dz4c_dt = dZ4cdt_prd["p1"] + dZ4cdt_prd["p2"] + dZ4cdt_prd["p3"] + dZ4cdt_prd["p4"] + dZ4cdt_prd["z3"] + dZ4cdt_prd["z5"] + dZ4cdt_prd["z6"] - dZ3cdt_prd["z4"] - dZ4cdt_rel_r1c - dZ4cdt_rel_r6c - dZ4cdt_rsp_o3c
    dz4n_dt = p1n_p1c*dZ4cdt_prd["p1"] + p2n_p2c*dZ4cdt_prd["p2"] + p3n_p3c*dZ4cdt_prd["p3"] + p4n_p4c*dZ4cdt_prd["p4"] + z3n_z3c*dZ4cdt_prd["z3"] + z5n_z5c*dZ4cdt_prd["z5"] + z6n_z6c*dZ4cdt_prd["z6"] - z4n_z4c*dZ3cdt_prd["z4"] - dZ4ndt_rel_r1n - dZ4ndt_rel_r6n - dZ4ndt_rel_n4n
    dz4p_dt = p1p_p1c*dZ4cdt_prd["p1"] + p2p_p2c*dZ4cdt_prd["p2"] + p3p_p3c*dZ4cdt_prd["p3"] + p4p_p4c*dZ4cdt_prd["p4"] + z3p_z3c*dZ4cdt_prd["z3"] + z5p_z5c*dZ4cdt_prd["z5"] + z6p_z6c*dZ4cdt_prd["z6"] - z4p_z4c*dZ3cdt_prd["z4"] - dZ4pdt_rel_r1p - dZ4pdt_rel_r6p - dZ4pdt_rel_n1p
    
    # microzooplankton
    dz5c_dt = dZ5cdt_prd["b1"] + dZ5cdt_prd["p1"] + dZ5cdt_prd["p2"] + dZ5cdt_prd["p3"] + dZ5cdt_prd["p4"] + dZ5cdt_prd["z6"] - dZ3cdt_prd["z5"] - dZ4cdt_prd["z5"] - dZ6cdt_prd["z5"] - dZ5cdt_rel_r1c - dZ5cdt_rel_r6c - dZ5cdt_rsp_o3c
    dz5n_dt = bn_bc*dZ5cdt_prd["b1"] + p1n_p1c*dZ5cdt_prd["p1"] + p2n_p2c*dZ5cdt_prd["p2"] + p3n_p3c*dZ5cdt_prd["p3"] + p4n_p4c*dZ5cdt_prd["p4"] + z6n_z6c*dZ5cdt_prd["z6"] - z5n_z5c*dZ3cdt_prd["z5"] - z5n_z5c*dZ4cdt_prd["z5"] - z5n_z5c*dZ6cdt_prd["z5"] - dZ5ndt_rel_r1n - dZ5ndt_rel_r6n - dZ5ndt_rel_n4n
    dz5p_dt = bp_bc*dZ5cdt_prd["b1"] + p1p_p1c*dZ5cdt_prd["p1"] + p2p_p2c*dZ5cdt_prd["p2"] + p3p_p3c*dZ5cdt_prd["p3"] + p4p_p4c*dZ5cdt_prd["p4"] + z6p_z6c*dZ5cdt_prd["z6"] - z5p_z5c*dZ3cdt_prd["z5"] - z5p_z5c*dZ4cdt_prd["z5"] - z5p_z5c*dZ6cdt_prd["z5"] - dZ5pdt_rel_r1p - dZ5pdt_rel_r6p - dZ5pdt_rel_n1p
    
    dz6c_dt = dZ6cdt_prd["b1"] + dZ6cdt_prd["p1"] + dZ6cdt_prd["p2"] + dZ6cdt_prd["p3"] + dZ6cdt_prd["p4"] + dZ6cdt_prd["z5"] - dZ3cdt_prd["z6"] - dZ4cdt_prd["z6"] - dZ5cdt_prd["z6"] - dZ6cdt_rel_r1c - dZ6cdt_rel_r6c - dZ6cdt_rsp_o3c
    dz6n_dt = bn_bc*dZ6cdt_prd["b1"] + p1n_p1c*dZ6cdt_prd["p1"] + p2n_p2c*dZ6cdt_prd["p2"] + p3n_p3c*dZ6cdt_prd["p3"] + p4n_p4c*dZ6cdt_prd["p4"] + z5n_z5c*dZ6cdt_prd["z5"] - z6n_z6c*dZ3cdt_prd["z6"] - z6n_z6c*dZ4cdt_prd["z6"] - z6n_z6c*dZ5cdt_prd["z6"] - dZ6ndt_rel_r1n - dZ6ndt_rel_r6n - dZ6ndt_rel_n4n
    dz6p_dt = bp_bc*dZ6cdt_prd["b1"] + p1p_p1c*dZ6cdt_prd["p1"] + p2p_p2c*dZ6cdt_prd["p2"] + p3p_p3c*dZ6cdt_prd["p3"] + p4p_p4c*dZ6cdt_prd["p4"] + z5p_z5c*dZ6cdt_prd["z5"] - z6p_z6c*dZ3cdt_prd["z6"] - z6p_z6c*dZ4cdt_prd["z6"] - z6p_z6c*dZ5cdt_prd["z6"] - dZ6pdt_rel_r1p - dZ6pdt_rel_r6p - dZ6pdt_rel_n1p

    # DOM
    dr1c_dt = (dP1cdt_lys_r1c + dP2cdt_lys_r1c + dP3cdt_lys_r1c + dP4cdt_lys_r1c) + dBcdt_lys_r1c - dBcdt_upt_r1c + (dZ5cdt_rel_r1c + dZ6cdt_rel_r1c)
    dr1n_dt = (dP1ndt_lys_r1n + dP2ndt_lys_r1n + dP3ndt_lys_r1n + dP4ndt_lys_r1n) + (extra_n1 + extra_n2 + extra_n3 + extra_n4) + dBcdt_lys_r1n - dBcdt_upt_r1c*r1n_r1c + (dZ5ndt_rel_r1n + dZ6ndt_rel_r1n)
    dr1p_dt = (dP1pdt_lys_r1p + dP2pdt_lys_r1p + dP3pdt_lys_r1p + dP4pdt_lys_r1p) + (dP1pdt_upt_r1p + dP2pdt_upt_r1p + dP3pdt_upt_r1p + dP4pdt_upt_r1p) + dBcdt_lys_r1p - dBcdt_upt_r1c*r1p_r1c + (dZ5pdt_rel_r1p + dZ6pdt_rel_r1p)
    dr2c_dt = (dP1cdt_exu_r2c + dP2cdt_exu_r2c + dP3cdt_exu_r2c + dP4cdt_exu_r2c) - dBcdt_upt_r2c + dBcdt_rel_r2c
    dr3c_dt = dBcdt_rel_r3c - dBcdt_upt_r3c

    # POM
    dr6c_dt = (dP1cdt_lys_r6c + dP2cdt_lys_r6c + dP3cdt_lys_r6c + dP4cdt_lys_r6c) + dBcdt_lys_r6c - dBcdt_upt_r6c + (dZ3cdt_rel_r6c + dZ4cdt_rel_r6c + dZ5cdt_rel_r6c + dZ6cdt_rel_r6c)
    dr6n_dt = (dP1ndt_lys_r6n + dP2ndt_lys_r6n + dP3ndt_lys_r6n + dP4ndt_lys_r6n) + dBcdt_lys_r6n - dBcdt_upt_r6c*r6n_r6c + (dZ3ndt_rel_r6n + dZ4ndt_rel_r6n + dZ5ndt_rel_r6n + dZ6ndt_rel_r6n)
    dr6p_dt = (dP1pdt_lys_r6p + dP2pdt_lys_r6p + dP3pdt_lys_r6p + dP4pdt_lys_r6p) + dBcdt_lys_r6p - dBcdt_upt_r6c*r6p_r6c + (dZ3pdt_rel_r6p + dZ4pdt_rel_r6p + dZ5pdt_rel_r6p + dZ6pdt_rel_r6p)
    dr6s_dt = dP1sdt_lys_r6s - dr6sdt_rmn_n5s + (p1s_p1c*(dZ3cdt_prd["p1"] + dZ4cdt_prd["p1"] + dZ5cdt_prd["p1"] + dZ6cdt_prd["p1"]))

    # Dissolved inorganic carbon
    do3c_dt = (-dP1cdt_gpp_o3c + dP1cdt_rsp_o3c) + (-dP2cdt_gpp_o3c + dP2cdt_rsp_o3c) + (-dP3cdt_gpp_o3c + dP3cdt_rsp_o3c) + (-dP4cdt_gpp_o3c + dP4cdt_rsp_o3c) + dBcdt_rsp_o3c + dZ3cdt_rsp_o3c + dZ4cdt_rsp_o3c + dZ5cdt_rsp_o3c + dZ6cdt_rsp_o3c + do3cdt_air_sea_flux
    
    # Total alkalinity (from Alkalinity.F90)
    if pel_chem_parameters["calc_alkalinity"] and o3c>0.0:
        do3h_dt = -dn3n_dt + dn4n_dt
    else:
        do3h_dt = 0.0

    return [do2o_dt, dn1p_dt, dn3n_dt, dn4n_dt, do4n_dt, dn5s_dt, dn6r_dt, db1c_dt, db1n_dt, db1p_dt, 
            dp1c_dt, dp1n_dt, dp1p_dt, dp1l_dt, dp1s_dt, dp2c_dt, dp2n_dt, dp2p_dt, dp2l_dt, 
            dp3c_dt, dp3n_dt, dp3p_dt, dp3l_dt, dp4c_dt, dp4n_dt, dp4p_dt, dp4l_dt, dz3c_dt, dz3n_dt, dz3p_dt,
            dz4c_dt, dz4n_dt, dz4p_dt, dz5c_dt, dz5n_dt, dz5p_dt, dz6c_dt, dz6n_dt, dz6p_dt, dr1c_dt, dr1n_dt, dr1p_dt, 
            dr2c_dt, dr3c_dt, dr6c_dt, dr6n_dt, dr6p_dt, dr6s_dt, do3c_dt, do3h_dt]

if __name__ == '__main__':
    # Names of species in the system
    species_names = ['O2o', 'N1p', 'N3n', 'N4n', 'O4n', 'N5s', 'N6r', 'B1c', 'B1n', 'B1p',
                     'P1c', 'P1n', 'P1p', 'P1l', 'P1s', 'P2c', 'P2n', 'P2p', 'P2l',
                     'P3c', 'P3n', 'P3p', 'P3l', 'P4c', 'P4n', 'P4p', 'P4l',
                     'Z3c', 'Z3n', 'Z3p', 'Z4c', 'Z4n', 'Z4p', 'Z5c', 'Z5n', 'Z5p',
                     'Z6c', 'Z6n', 'Z6p', 'R1c', 'R1n', 'R1p', 'R2c', 'R3c', 'R6c',
                     'R6n', 'R6p', 'R6s', 'O3c', 'O3h']

    # Initial concentrations
    c0 = [300.0,                    # O2o
          1.0,                      # N1p
          5.0,                      # N3n
          1.0,                      # N4n
          200.0,                    # O4n
          8.0,                      # N5s
          1.0,                      # N6r
          1.0,                      # B1c
          1.67e-2,                  # B1n
          1.85e-3,                  # B1p
          1.0,                      # P1c
          1.26e-2,                  # P1n
          7.86e-4,                  # P1p
          2.50e-2,                  # P1l
          1.00e-2,                  # P1s
          1.0,                      # P2c
          1.26e-2,                  # P2n
          7.86e-4,                  # P2p
          1.50e-2,                  # P2l
          1.0,                      # P3c
          1.26e-2,                  # P3n
          7.86e-4,                  # P3p
          2.00e-2,                  # P3l
          1.0,                      # P4c
          1.26e-2,                  # P4n
          7.86e-4,                  # P4p
          2.00e-2,                  # P4l
          1.0,                      # Z3c
          1.5e-2,                   # Z3n
          1.67e-3,                  # Z3p
          1.0,                      # Z4c
          1.5e-2,                   # Z4n
          1.67e-3,                  # Z4p
          1.0,                      # Z5c
          1.67e-2,                  # Z5n
          1.85e-3,                  # Z5p
          1.0,                      # Z6c
          1.67e-2,                  # Z6n
          1.85e-3,                  # Z6p
          1.0,                      # R1c
          1.26e-2,                  # R1n
          7.862e-4,                 # R1p
          0.1,                      # R2c
          1.0,                      # R3c
          1.0,                      # R6c
          1.26e-2,                  # R6n
          7.862e-4,                 # R6p
          1.45e-2,                  # R6s
          27060.0,                  # O3c
          2660.0                    # O3h
          ]

    # Time span for integration
    t_span = [0, 86400*365*10]

    # Assign multiplier = 1, indicating all species are present
    multiplier = numpy.ones(len(c0))

    # Integrate
    solution_full_model = solve_ivp(bfm50_rate_eqns, t_span, c0, method='RK23')
    
    # -------------------------------- Plots --------------------------------------
    # Plot style
    plt.rc('font', family='serif', size=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12, linewidth=2)
    
    species_names_units = ['Oxygen (mmol O$_2$/m$^3$)', 
                           'Phosphate (mmol P/m$^3$)',
                           'Nitrate (mmol N/m$^3$)', 
                           'Ammonium (mmol N/m$^3$)',
                           'Nitrogen sink (mmol N/m$^3$)', 
                           'Silicate (mmol Si/m$^3$)',
                           'Reduction Equivalents (mmol S/m$^3$)',
                           'Pelagic Bacteria (mg C/m$^3$)',
                           'Pelagic Bacteria (mmol N/m$^3$)',
                           'Pelagic Bacteria (mmol P/m$^3$)',
                           'Diatoms (mg C/m$^3$)',
                           'Diatoms (mmol N/m$^3$)',
                           'Diatoms (mmol P/m$^3$)',
                           'Diatoms (mg Chl-a/m$^3$)',
                           'Diatoms (mmol Si/m$^3$)',
                           'Flagellates (mg C/m$^3$)',
                           'Flagellates (mmol N/m$^3$)',
                           'Flagellates (mmol P/m$^3$)',
                           'Flagellates (mg Chl-a/m$^3$)',
                           'PicoPhytoplankton (mg C/m$^3$)',
                           'PicoPhytoplankton (mmol N/m$^3$)',
                           'PicoPhytoplankton (mmol P/m$^3$)',
                           'PicoPhytoplankton (mg Chl-a/m$^3$)',
                           'Large Phytoplankton (mg C/m$^3$)',
                           'Large Phytoplankton (mmol N/m$^3$)',
                           'Large Phytoplankton (mmol P/m$^3$)',
                           'Large Phytoplankton (mg Chl-a/m$^3$)',
                           'Carnivorous Mesozooplankton  (mg C/m$^3$)',
                           'Carnivorous Mesozooplankton  (mmol N/m$^3$)',
                           'Carnivorous Mesozooplankton  (mmol P/m$^3$)',
                           'Omnivorous Mesozooplankton (mg C/m$^3$)',
                           'Omnivorous Mesozooplankton (mmol N/m$^3$)',
                           'Omnivorous Mesozooplankton (mmol P/m$^3$)',
                           'Microzooplankton (mg C/m$^3$)',
                           'Microzooplankton (mmol N/m$^3$)',
                           'Microzooplankton (mmol P/m$^3$)',
                           'Heterotrophic Nanoflagellates (mg C/m$^3$)',
                           'Heterotrophic Nanoflagellates (mmol N/m$^3$)',
                           'Heterotrophic Nanoflagellates (mmol P/m$^3$)',
                           'Labile Dissolved Organic Carbon (mg C/m$^3$)',
                           'Labile Dissolved Organic Nitrogen (mmol N/m$^3$)',
                           'Labile Dissolved Organic Phosphate (mmol P/m$^3$)',
                           'Semi-labile Dissolved Organic Carbon (mg C/m$^3$)',
                           'Semi-refractory Dissolved Organic Carbon (mg C/m$^3$)',
                           'Particulate Organic Carbon (mg C/m$^3$)',
                           'Particulate Organic Nitrate (mmol N/m$^3$)',
                           'Particulate Organic Phosphate (mmol P/m$^3$)',
                           'Particulate Organic Silicate (mmol Si/m$^3$)',
                           'Dissolved Inorganic Carbon (mg C/m$^3$)',
                           'Total Alkalinity (mmol Eq/m$^3$)'
                           ]

    plt.close('all')
    with PdfPages('BFM50_plot.pdf') as pdf:
        for index,species in enumerate(species_names_units):
            fig = plt.figure(index)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.grid(linestyle='--')
            ax.plot(solution_full_model.t/86400/365, solution_full_model.y[index])
            ax.set_ylabel(species)
            ax.set_xlabel('Time (year)')
            ax.set_xlim(0,10)
            if index == 4:
                plt.ylim([199.9,200.1])
            if index == 42:
                plt.ylim([0.09,0.11])
            pdf.savefig(fig, bbox_inches = "tight")
            if index == 19 or index == 39 or index == 49:
                plt.close('all')
    plt.close('all')
