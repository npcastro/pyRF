#include <Python.h>

#include "prob.h"

static PyObject* pyRF_prob_cdf(PyObject* self, PyObject *args)
{
	double x;
	double mu;
	double s;
	double a;
	double b;

   if (!PyArg_ParseTuple(args, "fffff", &x, &mu, &s, &a, &b)) {
     double cdf;
     //cdf = normal_truncated_ab_cdf(x, mu, s, a, b);
   	 return Py_BuildValue('d', cdf);
   }
   
   /* Do something interesting here. */
   Py_RETURN_NONE;
}

static char prob_docs[] =
    "cdf(pivote, mean, std, l, r):  Determina la distribucion de probabilidad gaussiana acumulada entre dos bordes\npivote: punto a evaluar en la gaussiana\nmean: media de la gaussiana\nstd: desviacion standard\nl: limite izquierdo\nr: limite derecho\n";

static PyMethodDef pyRF_funcs[] = {
    {"cdf", (PyCFunction)pyRF_prob_cdf, 
     METH_VARARGS, prob_docs},
    { NULL, NULL, 0, NULL }
};

void initpyRF_prob(void)
{
    Py_InitModule3("pyRF_prob", pyRF_funcs,
                   prob_docs);
}