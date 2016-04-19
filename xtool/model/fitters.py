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
            bounds.append(getattr(self.model, parameter_name).bounds)
        return np.array(bounds)

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


    def fit_scipy_minimize(self, method, **kwargs):
        bounds = self._generate_bounds()
        starting_guess = self._get_starting_guess()
        start_time = time.time()

        result = optimize.minimize(
            self._calculate_chi2, starting_guess, method=method,
            bounds=bounds, **kwargs)
        print "Fit finished in {0} s".format(time.time() - start_time)

        for param_name, param_value in zip(self.model.fittable_parameter_names,
                                           result['x']):
            setattr(self.model, param_name, param_value)
        self.model.set_matrix_uncertainties(self.order)
        return result

    def _get_starting_guess(self):
        starting_guess = []
        for param_name in self.model.fittable_parameter_names:
            starting_guess.append(getattr(self.model, param_name).value)
        return np.array(starting_guess)


    def fit_least_squares(self, **kwargs):
        bounds = self._generate_bounds()
        starting_guess = self._get_starting_guess()

        result = optimize.least_squares(
            self._calculate_residuals, starting_guess, bounds=bounds.T,
            **kwargs)

        for param_name, param_value in zip(self.model.fittable_parameter_names,
                                           result['x']):
            setattr(self.model, param_name, param_value)
        return result

    def _calculate_chi2(self, param_values):
        call_param_dict = {key:value for key, value in zip(
            self.model.fittable_parameter_names, param_values)}
        return self.model.evaluate_chi2(self.order, **call_param_dict)

    def _calculate_residuals(self, param_values):
        call_param_dict = {key: value for key, value in zip(
            self.model.fittable_parameter_names, param_values)}
        return self.model.evaluate_residuals(self.order, **call_param_dict)


