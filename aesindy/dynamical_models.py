from .sindy_utils import library_size
import numpy as np

def get_model(name, args=None, normalization=None, use_sine=False):

    
    if name == 'lorenz':
        args = np.array([10, 28, 8/3]) if args is None else np.array(args)
        f = lambda z, t: [args[0]*(z[1] - z[0]), 
                        z[0]*(args[1] - z[2]) - z[1], 
                        z[0]*z[1] - args[2]*z[2]]

        dim = 3
        n = normalization if normalization is not None else np.ones((dim,))
        poly_order = 2 

        Xi = np.zeros((library_size(dim, poly_order), dim))
        Xi[1,0] = - args[0] 
        Xi[2,0] = args[0]*n[0]/n[1]
        Xi[1,1] = args[1]*n[1]/n[0]
        Xi[2,1] = -1
        Xi[6,1] = -n[1]/(n[0]*n[2])
        Xi[3,2] = -args[2]
        Xi[5,2] = n[2]/(n[0]*n[1])

        z0_mean_sug = [0, 0, 25]
        z0_std_sug = [36, 48, 41]

        
        
    elif name == 'rossler':
        args = [0.2, 0.2, 5.7] if args is None else np.array(args)
        f = lambda z, t: [-z[1] -  z[2] ,
                        z[0] + args[0]*z[1],
                        args[1] + z[2]*(z[0] - args[2])]
        dim = 3
        n = normalization if normalization is not None else np.ones((dim,))
        poly_order = 2 

        Xi = np.zeros((library_size(dim, poly_order), dim))
        Xi[2,0] = -n[0]/n[1] 
        Xi[3,0] = -n[0]/n[2] 
        Xi[1,1] = n[1]/n[0] 
        Xi[2,1] = args[0] 
        Xi[0,2] = n[2]*args[1] 
        Xi[3,2] = -args[2] 
        Xi[6,2] = 1.0/n[0]

        z0_mean_sug = [0, 1, 0]
        z0_std_sug = [2, 2, 2]

        

    elif name == 'predator_prey':
        args = [1.0, 0.1, 1.5, 0.75] if args is None else np.array(args)
        f = lambda z, t: [args[0]*z[0] - args[1]*z[0]*z[1] ,
                        -args[2]*z[1] + args[1]*args[3]*z[0]*z[1] ]
        dim = 2
        n = normalization if normalization is not None else np.ones((dim,))
        poly_order = 2 
        Xi = np.zeros((library_size(dim, poly_order), dim))
        Xi[1,0] = args[0] 
        Xi[4,0] = -args[1] * n[0]/n[1]
        Xi[2,1] = -args[2] 
        Xi[4,1] = args[1] * args[3] * n[1]/n[0]

        z0_mean_sug = [10, 5]
        z0_std_sug = [8, 8]

        
    elif name == 'pendulum':
        # Not easily renormalizable because f the sin(x) feature
        # g, L,
        args = [9.8, 1] if args is None else np.array(args)
        f = lambda z, t: [z[1],
                        -args[1]/args[0]*np.sin(z[0])]
        dim = 2
        n = normalization if normalization is not None else np.ones((dim,))
        poly_order = 1 
        use_sine = True  

        Xi = np.zeros((library_size(dim, poly_order, use_sine=use_sine), dim))
        Xi[2, 0] = 1
        Xi[3, 1] = -args[1]/args[0]

        z0_mean_sug = [np.pi/2, 0]
        z0_std_sug = [np.pi/2, 2]

    elif name == 'warfarin':
        # Args order: [CL, V1, Q, V2, kon, koff, kdeg, ksyn]
        # Default values from your config
        defaults = [0.1, 1.0, 0.5, 2.0, 0.01, 0.005, 0.001, 0.002]
        args = np.array(defaults) if args is None else np.array(args)
        
        # Unpack parameters for readability
        CL, V1, Q, V2, kon, koff, kdeg, ksyn = args

        # Define the ODE system (Vector Field)
        # z[0]=x1 (Central), z[1]=x2 (Periph), z[2]=x3 (Bound), z[3]=x4 (Target)
        f = lambda z, t: [
            -(CL/V1)*z[0] - (Q/V1)*z[0] + (Q/V2)*z[1] - kon*z[0]*z[3] + koff*z[2], # dx1/dt
            (Q/V1)*z[0] - (Q/V2)*z[1],                                             # dx2/dt
            kon*z[0]*z[3] - koff*z[2] - kdeg*z[2],                                 # dx3/dt
            -kon*z[0]*z[3] + koff*z[2] + ksyn - kdeg*z[3]                          # dx4/dt
        ]

        dim = 4
        poly_order = 2 
        # Normalization factor (handles scaling if you compress data range)
        n = normalization if normalization is not None else np.ones((dim,))

        # --- Construct Ground Truth SINDy Library (Xi) ---
        # Library order for 4 vars: 
        # [1, z0, z1, z2, z3, z0^2, z0z1, z0z2, z0z3, z1^2, ...]
        # Indices: 0, 1..4, 5..8 (z0 interactions)
        
        Xi = np.zeros((library_size(dim, poly_order), dim))
        
        # Equation 1: dx1/dt (Target: Index 0)
        Xi[1, 0] = -(CL/V1 + Q/V1)      # coeff of z0
        Xi[2, 0] = (Q/V2) * n[0]/n[1]   # coeff of z1
        Xi[3, 0] = koff * n[0]/n[2]     # coeff of z2 (x3)
        Xi[8, 0] = -kon * n[0]/(n[0]*n[3]) # coeff of z0*z3 (x1*x4) -- Index 8 is z0*z3

        # Equation 2: dx2/dt (Target: Index 1)
        Xi[1, 1] = (Q/V1) * n[1]/n[0]   # coeff of z0
        Xi[2, 1] = -(Q/V2)              # coeff of z1

        # Equation 3: dx3/dt (Target: Index 2)
        Xi[3, 2] = -(koff + kdeg)       # coeff of z2
        Xi[8, 2] = kon * n[2]/(n[0]*n[3]) # coeff of z0*z3

        # Equation 4: dx4/dt (Target: Index 3)
        Xi[0, 3] = ksyn * n[3]          # coeff of 1 (Constant)
        Xi[3, 3] = koff * n[3]/n[2]     # coeff of z2
        Xi[4, 3] = -kdeg                # coeff of z3
        Xi[8, 3] = -kon * n[3]/(n[0]*n[3]) # coeff of z0*z3

        # Initial Conditions (Mean and Std Dev for random sampling)
        z0_mean_sug = [10.0, 0.0, 0.0, 5.0]
        z0_std_sug = [1.0, 0.1, 0.1, 1.0] # Small variance for training diversity

    else: 
        raise ValueError(f"Model name '{name}' not found! Check spelling in params['model'].")       

    return f, Xi, dim, z0_mean_sug, z0_std_sug
