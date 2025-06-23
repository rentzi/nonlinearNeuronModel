import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy import signal


def ReLU(x):
    """
    Performs ReLU on a scalar or vector input

    Parameters
    ----------
    x: scalar or numpy array
        Input to the ReLU function
    
    Returns
    ----------
        Output of the ReLU function
    """

    return x * (x > 0)


def sigmoidR(x):
    """
    Output of a sigmoid function with a range between -1 and 1

    Parameters
    ----------
    x: scalar or numpy array
        Input to the sigmoid function
    
    Returns
    ----------
        Output of the sigmoid function
    """
    
    return ((2.0/(1.0+np.exp(-x))) - 1)


def XORNonlinearity(x,norm,maxV):
    """
    A lobe function representing the XOR nonlinearity which can have as an input a scalar or vector 

    Parameters
    ----------
    x: scalar or numpy array
        Input to the XOR function
    norm: scalar
        Half-width of the XOR lobe
    maxV: scalar
        x-value for which the center of the lobe (peak) is located
    
    Returns
    ----------
    sout: scalar or numpy array
        Output of the XOR function
    """

    inpt = np.absolute(x)
    exponent = ((inpt - maxV)/norm)**2    
    sout = np.exp(-exponent)
   
    return sout


def getGabor(c,f,theta,N,phase,std,gamma = 0.125):
    """
    Creates a Gabor numpy array 

    Parameters
    ----------
    c: scalar
        Input to the XOR function
    f: scalar
        Spatial frequency in cycles per image size
    theta: scalar
        Orientation of Gabor in radians
    N: scalar
        Size of the Gabor array (NxN)
    phase: scalar
        Phase of Gabor in radians
    std: scalar
        The standard deviation of the Gaussian envelope      
    gamma: scalar
        The aspect ratio of the Gaussian envelope

    Returns
    ----------
    gb: numpy array
        An NxN Gabor
    """
    
    theta = theta - np.pi/2
    sigmaX = std;sigmaY = std/gamma
    
    wavelength = 1/f
    
    x, y = np.meshgrid(np.arange(-np.fix(N/2),np.fix(N/2)+1),np.arange(np.fix(N/2),-np.fix(N/2)-1,-1) , indexing='xy')
    # x (right +)
    # y (up +)
    
    xTheta=x*np.cos(theta)+y*np.sin(theta)
    yTheta = -x*np.sin(theta)+y*np.cos(theta)
    gb=c*np.exp(-0.5*(((xTheta**2)/(sigmaX**2))+((yTheta**2)/(sigmaY**2))))*np.sin(2*(np.pi/wavelength)*xTheta+phase)
        
    return gb


def feedforwardModel(stim,w0,w1,lamda,lr,alpha,pa2m,flagNonlin = 0,maxV=0.43**0.5,XORNorm=0.2,maxminIterTuple = (21, 20)):
    """
    Runs the model with the first two terms, the linear and the XOR nonlinearity 

    Parameters
    ----------
    stim: numpy array
        Input stimulus to the model
    w0: numpy array
        The filter of the first linear term of the model
    w1: numpy array
        The filter of the second nonlinear term corresponding to the XOR nonlinearity
    lamda: scalar
        the unormalized weight of the XOR nonlinear term
    lr: scalar
        The learning rate
    alpha: scalar
        The attenuation of the backpropagating response      
    pa2m: scalar
        Exponent for conversion from activity (mean firing rate) to membrane potential
    flagNonlin: scalar
        If zero, the nonlinearity of the second term is an XOR, otherwise, it is a sigmoid
    maxV: numpy array
        x-value for which the center of the lobe (peak) is located in the XOR nonlinearity
    XORNorm: scalar
        Half-width of the XOR lobe in the XOR nonlinearity
    maxminIterTuple: tuple
        The maximum and minimum number of iterations of the model

    Returns
    ----------
    JNew: scalar
        The response of the model
    R: scalar
        The nonlinear response from the second (XOR) term
    Q: scalar
        The linear response
    """

    eps = 1e-2
    maxIterations,minIterations = maxminIterTuple
    JInit = 0.0;JOld = JInit
    
    n=1/(1+np.absolute(lamda))
    for iter_ in range(maxIterations):
   
        #linear part
        Q = np.sum(np.multiply(stim,w0))
       
        s2=(np.absolute(stim))**pa2m  #convert stim from spikes to voltage
        s00 = alpha*JOld
        if flagNonlin == 0:
            #xor nonlinearity
            outNonlin = XORNonlinearity(s2+s00,XORNorm,maxV)            
        else:
            #sigmoid nonlinearity
            outNonlin = sigmoidR(s2+s00)
        
        R = np.sum(np.multiply(w1,outNonlin))
        dJ = -JOld + n*Q + n*lamda*R
        JNew = JOld +lr*dJ                    
        
        #get the difference between the old and the new J
        diff = np.absolute(JNew - JOld)
        JOld = np.copy(JNew)

        if iter_ > minIterations and diff <eps:
            break
           
    return JNew,R,Q


def fullModel(stim, w0Dict, w1Dict, orientations, lamdas,lr, alphas,pa2m,maxV = 0.43**0.5,XORNorm=0.2, 
            maxminIterTuple = (21,20)):

    """
    Runs the model with all three terms. The difference with the feedforwardModel is that this implementation also includes a 
    third term where different units interact with each other. Therefore, we also have the response of more than one units 

    Parameters
    ----------
    stim: numpy array
        Input stimulus to the model
    w0Dict: dictionary
        w0Dict[ori] --> The filter of the first linear term of the model for the unit with preferred orientation ori 
    w1Dict: numpy array
        w1Dict[ori] --> The filter of the XOR nonlinear term of the model for the unit with preferred orientation ori
    orientations: numpy array
        Preferred orientation of each of the filters corresponding to a different unit. They are used as keys in the dictionaries above 
    lamdas: tuple
        the unormalized weights of the two nonlinear terms: the XOR (second), and the sigmoid (third)
    lr: scalar
        The learning rate
    alphas: tuple
        The attenuations of the backpropagating response in the second and third nonlinear terms      
    pa2m: scalar
        Exponent for conversion from activity (mean firing rate) to membrane potential
    maxV: numpy array
        x-value for which the center of the lobe (peak) is located in the XOR nonlinearity
    XORNorm: scalar
        Half-width of the XOR lobe in the XOR nonlinearity
    maxminIterTuple: tuple
        The maximum and minimum number of iterations of the model

    Returns
    ----------
    JNew: numpy array
        The response of the model for each unit
    R: numpy array
        The nonlinear response from the second (XOR) term for each unit
    Q: numpy array
        The linear response for each unit
    """
    
    iterOnsetRec = 1; eps = 1e-2
    maxIterations,minIterations = maxminIterTuple
    (lamda1,lamda2) = lamdas #(lamda of the second and third term)
    (alpha1,alpha2) = alphas # alpha1 (feedforward) and alpha2 (lateral)
    
    #calculation of weight matrix, w2
    numNeurons = len(w0Dict); w2Unorm = np.zeros((numNeurons,numNeurons))
    for i,k_i in enumerate(w0Dict.keys()):
        for j,k_j in enumerate(w0Dict.keys()):
            w2Unorm[i,j] = np.abs(np.sum(w0Dict[k_i]*w0Dict[k_j]))        
    
    normW2 = np.sum(w2Unorm,axis=1); normW2 = normW2[:,np.newaxis]
    w2 = numNeurons*(w2Unorm/normW2) 

    JInit = np.zeros(numNeurons); JOld = JInit 

    #Q does not change across iterations so we precompute it
    Q = np.zeros(numNeurons)
    for neuron in np.arange(numNeurons):
        #np.multipy does element by element mult, the * operator does matrix multiplication
        Q[neuron] = np.sum(np.multiply(stim,w0Dict[orientations[neuron]])) 
    
    JNew = np.zeros(numNeurons)
    # normalization full model , and feedforward (for the first iteration) 
    nPR = 1/(1+np.absolute(lamda1)+np.absolute(lamda2))
    nP = 1/(1+np.absolute(lamda1))
    
    for iter_ in range(maxIterations):
        for neuron in np.arange(numNeurons):
        
            #########for each iteration and each neuron 
            s00 = alpha1*JOld[neuron]
            
            s2=(np.absolute(stim))**pa2m
            outNonlin = XORNonlinearity(s2+s00,XORNorm,maxV)
            R = np.sum(np.multiply(w1Dict[orientations[neuron]],outNonlin))
            
            if iterOnsetRec <= iter_:
                RecDiff = JOld - alpha2*JOld[neuron]

                RecNonlin = sigmoidR(RecDiff)
                RecNonlinSum  = np.sum(w2[neuron,:]*RecNonlin)
                dJ = -JOld[neuron] + nPR*Q[neuron] + nPR*lamda1*R - nPR*lamda2*RecNonlinSum

            else:  
                dJ = -JOld[neuron] + nP*Q[neuron] + nP*lamda1*R
                 
            JNew[neuron] = JOld[neuron] +lr*dJ 
        
        
        #get the difference between the old and the new J
        diff = np.absolute(JNew - JOld)
        JOld = np.copy(JNew)
        if iter_ > minIterations and np.sum(diff) <eps:
            break
       
        
    return JNew,R,Q



# Get the parametric function from which you will get the best k value with curve_fit
def curveSNDN(x,k):
    """
    The parametric function used in online methods Eq 7 of "Fournier, et al. Nature neuroscience 14.8 (2011): 1053-1060."
    to quantify the relationship between the responses of neurons to sparse and dense noise stimuli. We use it in conjunction 
    to curve_fit (see Experiment3 jupyter file) to get the k value that gives the best fit with the data 

    Parameters
    ----------
    x: numpy array
        Input
    k: scalar
        Parameter of the function
    
    Returns
    ----------
        Output of the parametric function
    """   
    
    return x/(x+(((1/k)**2)*(1-x)))


def getDenseNoise(N):
    """
    Creates a dense noise stimulus by randomly interspersing gray, white or black elements with an equal probability 

    Parameters
    ----------
    N: scalar
        Size of the dense noise array (NxN)

    Returns
    ----------
    s: numpy array
        An NxN dense noise stimulus
    """    
    s_ = np.random.uniform(size=(N,N))
    s = np.round(2*s_)
    return s


def getSparseNoise(N,ns):
    """
    Creates a sparse noise stimulus by randomly interspersing gray, white or black elements with the number of elements that 
    contributed to the noise (white or black) being ns 

    Parameters
    ----------
    N: scalar
        Size of the sparse noise array (NxN)
    ns: scalar
        number of sparse noise elements randomly interspersed in the image

    Returns
    ----------
    s: numpy array
        An NxN sparse noise stimulus
    """  
    
    s=np.ones((N,N))
    for j in np.arange(1,ns):
        row=int(np.ceil((N-1)*np.random.uniform(size = 1)))
        col=int(np.ceil((N-1)*np.random.uniform(size = 1)))
        s[col,row]=np.round(2*np.random.uniform(size = 1))    
    
    return s



def XORPrime(x,norm,maxV):
    """
    The derivative of the XORNonlinearity which can have as an input a scalar or vector 

    Parameters
    ----------
    x: scalar or numpy array
        Input to the XOR function
    norm: scalar
        Half-width of the XOR lobe
    maxV: scalar
        x-value for which the center of the lobe (peak) is located
    
    Returns
    ----------
    sout: scalar or numpy array
        Output of the XOR function
    """

    inpt = np.absolute(x)
    exponent = ((inpt - maxV)/norm)**2
    part = ((inpt - maxV))*(-2/norm)
    
    soutP = np.exp(-exponent)*part    
    
    return soutP


def computeRF(stim,w0,w1, lamda, alpha,pa2m=0.5,lr=0.1):
    """
    Computes the RF of the model. The RF is the derivative of the model (we exclude the third intracortical term) wrt position (eq 10) 

    Parameters
    ----------
    stim: numpy array
        Input stimulus
    w0: numpy array
        The filter of the first linear term of the model
    w1: numpy array
        The filter of the second nonlinear term corresponding to the XOR nonlinearity
    lamda: scalar
        the unormalized weight of the XOR nonlinear term
    alpha: scalar
        The attenuation of the backpropagating response      
    pa2m: scalar
        Exponent for conversion from activity (mean firing rate) to membrane potential
    lr: scalar
        The learning rate

    Returns
    ----------
    RF: numpy array
        The spatial receptive field
    vi: scalar
        The response of the model
    """    
    
    eps = 1e-2
    maxV = 0.43**pa2m; XORNorm = 0.2; flagNonlin=0
    J,_,_   = feedforwardModel(stim,w0,w1,lamda,lr,alpha,pa2m,flagNonlin = flagNonlin, maxV=maxV)
    vi = J
    
    n=1/(1+np.absolute(lamda))
    c0 = n; c1 = lamda*n
    s2=(np.absolute(stim))**pa2m
    
    dxHatDx = pa2m*((eps+np.absolute(stim))**(pa2m-1))*np.sign(stim)
    
    sigma1p = XORPrime(s2+alpha*vi,XORNorm,maxV)
    
    part = c1*w1*sigma1p; numerator = (c0*w0) + (part*dxHatDx); denominator = 1 - part*alpha
    RF = numerator/denominator
    
    return RF, vi




def gratingH(sf,N):
    """
    Creates a horizontal grating  (varying along the vertical direction)

    Parameters
    ----------
    sf: scalar
        Spatial frequency in cycles per image size
    N: scalar
        Size of the grating (NxN)

    Returns
    ----------
    gH: numpy array
        An NxN horizontal grating
    """
    
    phase = np.pi/2
    f = sf/N
    gH = np.zeros((N,N))
    if (sf == 0):
        gH = np.ones((N,N))
    else:
        for i in range(N):
            for j in range(N):
                gH[i,j] = np.sin(2*np.pi*f*i + phase)
                
    
    return gH


def gratingV(sf,N):
    """
    Creates a vertical grating (varying along the horizontal direction) 

    Parameters
    ----------
    sf: scalar
        Spatial frequency in cycles per image size
    N: scalar
        Size of the grating (NxN)

    Returns
    ----------
    gV: numpy array
        An NxN vertical grating
    """    
    
    phase = np.pi/2
    f = sf/N
    gV = np.zeros((N,N))
    if (sf == 0):
        gV = np.ones((N,N))
    else:
        for i in range(N):
            for j in range(N):
                gV[i,j] = np.sin(2*np.pi*f*j + phase)
                
    
    return gV
