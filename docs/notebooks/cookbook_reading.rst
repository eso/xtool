
Reading Spectra with xtool
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from xtool.data import XShooterData, Order


.. parsed-literal::

    /Users/wkerzend/anaconda3/envs/extract/lib/python2.7/site-packages/IPython/kernel/__init__.py:13: ShimWarning: The `IPython.kernel` package has been deprecated. You should import from ipykernel or jupyter_client instead.
      "You should import from ipykernel or jupyter_client instead.", ShimWarning)


.. code:: python

    xd = XShooterData('xtool_ds') # with the absolute or relative path given
    xd




.. parsed-literal::

    <XShooterData XSHOO.2013-03-04T07:06:29.289.fits NIR>



You can access the individual orders given in the dataset by just indexing the dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    xd[11]




.. parsed-literal::

    <Order 11 XSHOO.2013-03-04T07:06:29.289.fits NIR>



.. code:: python

    xd[5]


::


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-6-dee1a2b27f4d> in <module>()
    ----> 1 xd[5]
    

    /Users/wkerzend/projects/eso/xtool/xtool/data.pyc in __getitem__(self, item)
         52         if item not in self.order_table.index:
         53             raise IndexError('Dataset only contains orders {0} - {1}'.format(
    ---> 54                 self.order_table.index.min(), self.order_table.index.max()))
         55         else:
         56             return Order.from_xshooter_data(self, item)


    IndexError: Dataset only contains orders 11 - 26


Each order contains the cutout of the order in masked arrays that cutout intraorder regions.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    order = xd[11]

Each one of the order objects also contains a WCS that can transform
between pixelspace and woorld coordinate system (angstrom, slit
position)

.. code:: python

    order.order_wcs(51, 52)




.. parsed-literal::

    (2474.587158203125, 0.5863650441169739)


