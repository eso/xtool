from scipy import optimize
import numpy as np
import time

class Fitter(object):
    def __init__(self, model, order):
        """
        Parameters
        ----------
        model : xtool.model.OrderModel
        order : xtool.data.Order
        """

        self.model = model
        self.order = order

    def _generate_bounds(self):
        """
        Generating bounds from the Model parameters

        Returns
        -------
        numpy.ndarray
            returns lower bounds on row 0 and upper bounds on row 1
        """

        bounds = []
        for parameter_name in self.model.fittable_parameter_names:
            if parameter_name == 'wave_transform_coef':
                bounds.append((np.nan, np.nan))
            else:
                bounds.append(getattr(self.model, parameter_name).bounds)
        return np.array(bounds)

    def _generate_starting_guess(self):
        """
        Generate Starting guesses

        Returns
        -------
            : np.ndarray
            array of starting guesses

        """
        starting_guess = []
        for param_name in self.model.fittable_parameter_names:
            if param_name == 'wave_transform_coef':
                starting_guess.append(
                    getattr(
                        self.model.virtual_pixel, param_name).value.flatten())
            else:
                starting_guess.append(getattr(self.model, param_name).value)
        param_shapes = []
        for item in starting_guess:
            try:
                len(item)
            except TypeError:
                param_shapes.append(1)
            else:
                param_shapes.append(len(item))

        return np.array(starting_guess), param_shapes



    def fit_differential_evolution(self, **kwargs):
        bounds = self._generate_bounds()
        start_time = time.time()
        result = optimize.differential_evolution(self._calculate_chi2, bounds,
                                                 **kwargs)
        print "Fit finished in {0} s".format(time.time() - start_time)

        for param_name, param_value in zip(self.model.fittable_parameter_names,
                                           result['x']):
            setattr(self.model, param_name, param_value)
        return result


    def fit_scipy_minimize(self, method, ylim=None, **kwargs):
        bounds = self._generate_bounds()
        starting_guess, param_shape = self._generate_starting_guess()
        start_time = time.time()

        result = optimize.minimize(
            self._calculate_chi2, starting_guess, method=method,
            bounds=bounds, args=(param_shape, ylim), **kwargs)
        print "Fit finished in {0} s".format(time.time() - start_time)

        for param_name, param_value in zip(self.model.fittable_parameter_names,
                                           result['x']):
            setattr(self.model, param_name, param_value)
        #self.model.set_matrix_uncertainties(self.order)
        return result


    def fit_least_squares(self, ylim=None, **kwargs):
        bounds = self._generate_bounds()
        starting_guess = self._generate_starting_guess()

        result = optimize.least_squares(
            self._calculate_residuals, starting_guess, bounds=bounds.T,
            **kwargs)


        print "change back this is only for test"
        return result

        for param_name, param_value in zip(self.model.fittable_parameter_names,
                                           result['x']):
            if param_name == 'wave_transform_coef':
                setattr(self.model, param_name, param_value)
            else:
                setattr(self.model, param_name, param_value)
        return result

    def _calculate_chi2(self, param_values, param_shape, ylim=None):
        call_param_dict = {}
        current_idx = 0
        for i, param_name in enumerate(self.model.fittable_parameter_names):

            param_len = param_shape[i]
            call_param_dict[param_name] = param_values[
                                          current_idx:current_idx + param_len]
            current_idx = current_idx + param_len

        chi2 = self.model.evaluate_chi2(self.order, ylim=ylim,
                                        **call_param_dict)
        return chi2

    def _calculate_residuals(self, param_values, ylim=None):
        call_param_dict = {key: value for key, value in zip(
            self.model.fittable_parameter_names, param_values)}
        return self.model.evaluate_residuals(self.order, ylim=ylim, **call_param_dict)


