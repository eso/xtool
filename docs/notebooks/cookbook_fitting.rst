
Extracting Spectra with xtool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from xtool.data import XShooterData, Order
    from xtool.model.base import OrderModel, GenericBackground, MoffatTrace, VirtualPixelWavelength
    
    from scipy import sparse
    from scipy import optimize

Reading XShooter data
~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    xd = XShooterData('xtool_ds/')

.. code:: python

    current_order = xd[17]

Generating a virtual pixel table for "Wavelength"-pixels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    virt_pix = VirtualPixelWavelength.from_order(current_order)
    pixel_table = virt_pix()

Initializing the two Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    background_mdl = GenericBackground(pixel_table)
    trace_mdl = MoffatTrace(pixel_table)

.. code:: python

    order_model = OrderModel([background_mdl, trace_mdl])

Show fittable parameters
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    order_model




.. parsed-literal::

    <OrderModel(background_level=[  4.65477892e+18   4.65477936e+18   4.65477892e+18 ...,   4.65419002e+18
       4.65419002e+18   4.65419002e+18], amplitude=[ nan  nan  nan ...,  nan  nan  nan], trace_pos=0.0, sigma=1.0, beta=1.5 [f])>



Change fittable parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    order_model.trace_pos




.. parsed-literal::

    Parameter('trace_pos', value=0.0)



.. code:: python

    order_model.trace_pos = 10.

.. code:: python

    order_model.trace_pos




.. parsed-literal::

    Parameter('trace_pos', value=10.0)



Generating a model
^^^^^^^^^^^^^^^^^^

1. We generate a design matrix
   (https://en.wikipedia.org/wiki/Design\_matrix)
2. We solve the design matrix

The evaluate does both of these steps at the same time

.. code:: python

    # Generating the design matrix often depicted as capital A
    
    A = order_model.generate_design_matrix(trace_pos=-5, sigma=1.5)
    
    # adding the uncertainties to the design matrix
    A.data /= current_order.uncertainty.compressed()[A.row]
    
    # making a vector of the result pixels often depicted as lower-case b
    
    b = current_order.data.compressed() / current_order.uncertainty.compressed()
    result = sparse.linalg.lsmr(A, b)

.. code:: python

    result




.. parsed-literal::

    (array([-139.75576407,  803.07479466,  734.99747033, ...,   36.2185333 ,
             139.07635082,  118.40347293]),
     2,
     1292,
     3229.9116167401157,
     0.078203559417645799,
     24.225493274011736,
     32.189878788183037,
     272679.62519832124)


