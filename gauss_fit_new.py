import numpy
import matplotlib.pyplot as plt
import copy
import os
from uncertainties import ufloat
import numpy as np
from lmfit import Model
from tqdm import tqdm
import scienceplots

def gaussian(x, amplitude, center, sigma):
    
    return amplitude * numpy.exp(-(x-center)**2 / (2*sigma**2))

def freq_to_velocity(freq_width, rest_freq):
    c = 299792.458
    return c * ((freq_width) / rest_freq)

def velocity_to_freq(velocity_width, rest_freq):
    c = 299792.458
    return velocity_width * rest_freq / c

def fit_spectrum_with_gaussians(obs_X, obs_Y, threshold, line_width_kms, max_iterations, rest_freq, debug=True):
    output_dir = 'intermediate_plots_lmfit'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class ModelResult:
        def __init__(self):
            self.obs_X = []
            self.obs_Y = []
            self.p = []
            self.err_p = []
            self.fit_quality = 0.0
            self.debug_info = {}
            self.params = None

    def estimate_initial_parameters_g(x, y, index_max, window_points, freq_width, previous_widths):
        # Select window around the maximum
        print(index_max)
        
        start_idx = max(0, index_max - window_points)
        end_idx = min(len(x) - 1, index_max + window_points)
        
        x_window = x[start_idx:end_idx+1]
        y_window = y[start_idx:end_idx+1]
        
        # Initial guesses
        amplitude_guess = y[index_max]
        center_guess = x[index_max]
        width_guess = freq_width
        
        # Create model and perform the fit
        gmodel = Model(gaussian)
        params = gmodel.make_params(amplitude=amplitude_guess, center=center_guess, sigma=width_guess)
        
        # Add bounds to parameters
        params['amplitude'].min = 0  # Amplitude should be positive
        #params['center'].min = center_guess - freq_width/5
        #params['center'].max = center_guess + freq_width/5
        params['sigma'].min = width_guess * 0.5
        params['sigma'].max = width_guess * 15.5
        
        try:
            result = gmodel.fit(y_window, params, x=x_window)
            amplitude = result.params['amplitude'].value
            center = result.params['center'].value
            sigma = result.params['sigma'].value
            
            return amplitude, center, sigma
        except Exception as e:
            if debug:
                print(f"Initial parameter estimation failed: {str(e)}")
            return amplitude_guess, center_guess, width_guess
    
    obs_X0 = copy.deepcopy(obs_X)
    obs_Y0 = copy.deepcopy(obs_Y)
    
    freq_width = abs(velocity_to_freq(line_width_kms, rest_freq))
    print(freq_width)
    
    noise_level = threshold
    
    models = []
    previous_widths = []
    working_obs_Y = copy.deepcopy(obs_Y)
    freq_step = numpy.abs(obs_X0[1] - obs_X0[0])
    window_points = int((freq_width) / freq_step) * 4
    peak_counter = 0
    
    gmodel = Model(gaussian)
    
    while True:
        index_of_maximum_flux = numpy.argmax(working_obs_Y)
        current_peak = working_obs_Y[index_of_maximum_flux]
        
        if current_peak < threshold or peak_counter >= max_iterations:
            break
            
        peak_counter += 1

        start_idx = max(0, index_of_maximum_flux - window_points)
        end_idx = min(len(obs_X) - 1, index_of_maximum_flux + window_points)
        
        # Get initial parameters
        amp_init, center_init, width_init = estimate_initial_parameters_g(
            obs_X, working_obs_Y, index_of_maximum_flux, window_points, freq_width, previous_widths)
            
        if peak_counter == 1:
            first_amp_init = amp_init
            first_center_init = center_init
            first_width_init = width_init    
        
        # Create a temporary model and prepare for fitting
        temp_model = ModelResult()
        temp_model.obs_X = obs_X[start_idx:end_idx+1]
        temp_model.obs_Y = working_obs_Y[start_idx:end_idx+1]
        
        try:
            # Create parameter set with constraints
            params = gmodel.make_params(
                amplitude=amp_init,
                center=center_init,
                sigma=first_width_init
            )
            
            # Set parameter bounds
            params['amplitude'].min = 0
            params['amplitude'].max = amp_init +amp_init*2.5
            params['center'].min = center_init - freq_width/1000
            params['center'].max = center_init + freq_width/1000
            params['sigma'].min = first_width_init*0.03
            params['sigma'].max = first_width_init * 3.0
            
            # Perform the fit
            result = gmodel.fit(temp_model.obs_Y, params, x=temp_model.obs_X)
            
            if result.success:
                # Store parameters and errors
                temp_model.params = result.params
                temp_model.p = [
                    result.params['amplitude'].value,
                    result.params['center'].value,
                    result.params['sigma'].value
                ]
                
                temp_model.err_p = [
                    result.params['amplitude'].stderr if result.params['amplitude'].stderr else result.params['amplitude'].stderr*(-1),
                    result.params['center'].stderr if result.params['center'].stderr else 0,
                    result.params['sigma'].stderr if result.params['sigma'].stderr else 0
                ]
                
                # Calculate fit quality (R-squared)
                ss_res = numpy.sum(result.residual**2)
                ss_tot = numpy.sum((temp_model.obs_Y - numpy.mean(temp_model.obs_Y))**2)
                temp_model.fit_quality = 1 - (ss_res / ss_tot)
                
                if temp_model.fit_quality > 0.05:
                    models.append(temp_model)
                    previous_widths.append(temp_model.p[2])
                    
                    # Subtract fitted Gaussian
                    fitted_gaussian = gaussian(obs_X0, temp_model.p[0], temp_model.p[1], temp_model.p[2]*1.3)
                    working_obs_Y -= fitted_gaussian
                    
                    if debug:
                        print(f"Successful fit: quality={temp_model.fit_quality:.3f}")
                        print(f"Parameters: A={temp_model.p[0]:.6f}, μ={temp_model.p[1]:.6f}, σ={temp_model.p[2]:.6f}")
                        print(f"Errors: ΔA={temp_model.err_p[0]:.6f}, Δμ={temp_model.err_p[1]:.6f}, Δσ={temp_model.err_p[2]:.6f}")
                else:
                    if debug:
                        print(f"Rejected: Low quality fit ({temp_model.fit_quality:.3f})")
                    working_obs_Y[index_of_maximum_flux] = 0
            else:
                if debug:
                    print(f"Fit failed to converge")
                working_obs_Y[index_of_maximum_flux] = 0
                
        except Exception as e:
            if debug:
                print(f"Fitting error: {str(e)}")
            working_obs_Y[index_of_maximum_flux] = 0
    
    # Plot and save results
    plot_final_results(obs_X0, obs_Y0, models, output_dir, rest_freq)
    save_results(models, rest_freq)
    
    return models

def plot_final_results(obs_X0, obs_Y0, models, output_dir, rest_freq):
    with plt.style.context(['science', 'no-latex']):
        
        # Create figure with GridSpec for complex layout
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 2, height_ratios=[2, 1, 1])
        
        # Main spectrum plot (top panel spanning both columns)
        ax_main = fig.add_subplot(gs[0, :])
        
        ax_main.step(obs_X0, obs_Y0, 'k-', label='Observations', linewidth=1, markersize=1)
        
        # Calculate total model and plot individual components
        total_model = numpy.zeros_like(obs_X0)
        for i, model in enumerate(models):
            gauss_curve = gaussian(obs_X0, model.p[0], model.p[1], model.p[2])
            ax_main.plot(obs_X0, gauss_curve, 'r-', alpha=0.7)
            total_model += gauss_curve
            
        # Plot threshold line
        r = numpy.full_like(obs_Y0, 0.00551 * 3)
        plt.plot(obs_X0, r, '--y')
        
        ax_main.set_xlabel('$\\nu$, GHz', fontsize=30)
        ax_main.set_ylabel('T$_{mb}$, K', fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        ax_main.legend(fontsize=25)
        
        # Residuals plot (middle panel)
        ax_residuals = fig.add_subplot(gs[1, :])
        residuals = obs_Y0 - total_model
        ax_residuals.plot(obs_X0, residuals, 'b-', label='Residuals')
        ax_residuals.axhline(y=0, color='r', linestyle=':')
        ax_residuals.set_xlabel('Frequency (GHz)')
        ax_residuals.set_ylabel('Residuals (K)')
        
        # Parameter correlation plot (bottom right)
        ax_corr = fig.add_subplot(gs[2,:])
        if len(models) > 0:
            # Extract parameters
            amplitudes = [m.p[0] for m in models]
            widths = [freq_to_velocity(abs(m.p[2]*2.355), rest_freq) for m in models]
            
            # Create scatter plot
            scatter = ax_corr.scatter(amplitudes, widths, c=widths, 
                                    cmap='viridis', alpha=0.6)
            plt.plot(amplitudes, widths, 'bo', label='Line width (km/s)')
            ax_corr.set_xlabel('Amplitude (K)')
            ax_corr.set_ylabel('Line width (km/s)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_fit_lmfit_diagnostic.png'), dpi=300)
        plt.show()
        plt.close()
        
        # Create additional diagnostic plots
        fig_diag = plt.figure(figsize=(12, 8))
        gs_diag = plt.GridSpec(2, 2)
        
        # Velocity width distribution
        ax_vel = fig_diag.add_subplot(gs_diag[0, 0])
        velocities = [freq_to_velocity(m.p[2]*2.355, rest_freq) for m in models]
        ax_vel.hist(velocities, bins=25, alpha=0.7)
        ax_vel.set_xlabel('Line width (km/s)')
        ax_vel.set_ylabel('Count')
        ax_vel.set_title('Distribution of Line Widths')
        
        # Amplitude distribution
        ax_amp = fig_diag.add_subplot(gs_diag[0, 1])
        amplitudes = [m.p[0] for m in models]
        ax_amp.hist(amplitudes, bins=15, alpha=0.7)
        ax_amp.set_xlabel('Amplitude (K)')
        ax_amp.set_ylabel('Count')
        ax_amp.set_title('Distribution of Amplitudes')
        
        # Signal-to-noise ratio
        ax_snr = fig_diag.add_subplot(gs_diag[1, 0])
        noise_level = 0.0001
        snr = [m.p[0]/noise_level for m in models]
        ax_snr.hist(snr, bins=15, alpha=0.7)
        ax_snr.set_xlabel('Signal-to-Noise Ratio')
        ax_snr.set_ylabel('Count')
        ax_snr.set_title('Distribution of SNR')
        
        # Fit quality
        ax_qual = fig_diag.add_subplot(gs_diag[1, 1])
        quality = [m.fit_quality for m in models]
        ax_qual.hist(quality, bins=15, alpha=0.7)
        ax_qual.set_xlabel('R-squared')
        ax_qual.set_ylabel('Count')
        ax_qual.set_title('Distribution of Fit Quality')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_fit_lmfit_statistics.png'), dpi=300)
        
        plt.close()

def save_results(models, rest_freq):
    c = 299792.458
    with open('/home/anastasiaf/RCW120/CORE2/CORE2_FREQ_RESAMPLED/baseline/CORE2_B11_res_freq_baseline_gauss.dat', 'w') as f:
        for i, model in enumerate(models):
            # Get parameters and errors directly from lmfit results
            amplitude = model.p[0]
            amp_err = model.err_p[0]
            center = model.p[1]
            center_err = model.err_p[1]
            sigma = model.p[2]
            sigma_err = model.err_p[2]
            
            # Use uncertainties package for error propagation
            
            vel1=freq_to_velocity(sigma, rest_freq)
            vel1_err=freq_to_velocity(sigma_err, rest_freq)
            # Calculate velocity width and its error
            vel = freq_to_velocity(sigma, rest_freq) * 2.355  # FWHM = 2.355 * sigma
            vel_ufloat = freq_to_velocity(sigma_err, rest_freq) * 2.355
            
            # Calculate area with proper error propagation
            # Area = amplitude * sigma * sqrt(2*pi)
            area = amplitude * vel1 * numpy.sqrt(2*numpy.pi)
            
            # Create ufloat for amplitude to propagate errors
            
            
            # Calculate area using ufloats to handle error propagation
            area_ufloat = numpy.sqrt(
                (vel1 * amp_err)**2 +  # derivative wrt amplitude
                (amplitude  * vel1_err)**2    # derivative wrt width
            )
            
            # Write to file
            f.write(f"Gaussian {i+1}:\n")
            f.write(f"  Amplitude = {abs(amplitude):.3f} ± {amp_err:.3f}\n")
            f.write(f"  Center = {center:.6f} ± {center_err:.6f}\n")
            f.write(f"  Width = {abs(vel):.3f} ± {abs(vel_ufloat):.3f} \n")
            f.write(f"  Area = {abs(area):.3f} ± {abs(area_ufloat):.3f}\n")
            f.write("-------------------------------\n")
            
            # Also print to console
            print(f"Line {i+1}:")
            print(f"  Amplitude = {abs(amplitude):.3f} ± {amp_err:.3f} K")
            print(f"  Center = {center:.6f} ± {center_err:.6f} GHz")
            print(f"  Width = {abs(vel):.3f} ± {abs(vel_ufloat):.3f} km/s")
            print(f"  Area = {abs(area):.3f} ± {abs(area_ufloat):.3f} K·km/s")
            print("-------------------------------")

if __name__ == "__main__":
    name = '/home/anastasiaf/RCW120/CORE2/CORE2_FREQ_RESAMPLED/baseline/CORE2_B5_res_freq_baseline.dat'
    obsX, obsY = numpy.loadtxt(name, unpack=True)
    obsX = obsX * 1e-3
    
    
    dv = 1  # Line width in km/s
    
   
    
    models = fit_spectrum_with_gaussians(
        obsX, obsY,
        threshold=0.001,
        line_width_kms=dv,
        max_iterations=1000,
        rest_freq=205.000988
    )
