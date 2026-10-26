# Complete_Machine_Learning

This Repo consist of all ML model and concept to Learn in Machine Learning

Chapter 4 Study Guide: Spatial Models Using Partial Differential Equations
1. Introduction to Spatial PDE Models
1.1 Ordinary Differential Equations (ODEs) vs. Partial Differential Equations (PDEs)
When modeling real-world phenomena, Ordinary Differential Equations (ODEs) restrict the analysis to a single independent variable—typically time ( $t$ ). ODE models assume that time is the sole dominant factor and ignore spatial variation. Consequently, a disparity between an ODE model's predictions and empirical data often indicates that state variables depend on more than one independent variable (e.g., time and space).
Partial Differential Equations (PDEs) overcome this restriction by incorporating partial derivatives with respect to two or more independent variables simultaneously (such as time  $t$  and spatial coordinates  $x, y, z$ ).
Example: Predator-Prey Dynamics (Lotka-Volterra)
Temporal ODE Model:   $$\frac{dP_1(t)}{dt} = \alpha P_1(t) - \beta P_1(t)P_2(t)$$   $$\frac{dP_2(t)}{dt} = -\gamma P_2(t) + \delta P_1(t)P_2(t)$$   Here,  $P_1(t)$  and  $P_2(t)$  represent prey and predator populations over time.
Spatial-Temporal PDE Model:  To account for spatial dispersion and physical movement, diffusion terms  $D_1 \nabla^2 P_1$  and  $D_2 \nabla^2 P_2$  are added:  $$\frac{\partial P_1(t, x, y)}{\partial t} = \alpha P_1(t, x, y) - \beta P_1(t, x, y)P_2(t, x, y) + D_1 \nabla^2 P_1(t, x, y)$$   $$\frac{\partial P_2(t, x, y)}{\partial t} = -\gamma P_2(t, x, y) + \delta P_1(t, x, y)P_2(t, x, y) + D_2 \nabla^2 P_2(t, x, y)$$  where  $\nabla^2 \cong \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$  is the Laplacian operator in two dimensions.
1.2 Classifications of Problems and Boundary Conditions
The general solution of a PDE contains as many arbitrary functions as the order of the highest partial differential coefficient. Determining these functions requires supplementary conditions defined over a spatial domain  $\Omega$ , its boundary  $\partial\Omega$ , or an initial time  $t$ .
Problem Types
Initial Value Problem (IVP):  A PDE coupled with initial conditions (ICs) prescribed at an initial point/time of the domain.
Boundary Value Problem (BVP):  A PDE coupled with boundary conditions (BCs) prescribed at the boundaries of the domain.
Initial Boundary Value Problem (IBVP):  A PDE requiring both initial conditions across the domain and boundary conditions along the domain boundary over time.
Categories of Boundary Conditions (BCs)
Dirichlet Boundary Condition:  The value of the dependent variable  $u$  is specified on the boundary.
Neumann Boundary Condition:  The normal derivative of the dependent variable  $\frac{\partial u}{\partial n}$  is specified on the boundary.
Cauchy Boundary Condition:  Both the value of the dependent variable  $u$  and its normal derivative  $\frac{\partial u}{\partial n}$  are specified on the boundary.
Robin Boundary Condition:  A linear combination of the dependent variable and its normal derivative (e.g.,  $c_1 u + c_2 \frac{\partial u}{\partial n}$ ) is specified on the boundary.
1.3 Classification of Second-Order PDEs
A general linear second-order partial differential equation in two independent variables has the form:  $$A u_{xx} + B u_{xy} + C u_{yy} + D u_x + E u_y + F u = 0$$
The classification depends strictly on the discriminant  $B^2 - 4AC$ :
Discriminant Value,PDE Classification,Boundary/Initial Condition Requirements
$B^2 - 4AC < 0$,Elliptic,Requires Dirichlet or Neumann boundary conditions on a  closed boundary  surrounding the region of interest. (Cauchy conditions on open surfaces yield instabilities or non-unique/overly restrictive solutions).
$B^2 - 4AC = 0$,Parabolic,Requires Dirichlet or Neumann boundary conditions on an  open surface .
$B^2 - 4AC > 0$,Hyperbolic,Requires Cauchy boundary conditions on an  open surface .
1.4 Problem Formulation Examples
IVP Example:   $$\frac{\partial^2 u}{\partial t^2} - \frac{\partial^2 u}{\partial x^2} = 0, \quad 0 \le x \le l, \, t > 0$$   $$\text{ICs: } u(x, 0) = 0, \quad \frac{\partial u(x, 0)}{\partial t} = x$$
BVP Examples (  $\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2} = 0, \, 0 \le x \le L, \, t > 0$  ):
Dirichlet:   $u(0, t) = 0, \quad u(L, t) = 0$
Neumann:   $\frac{\partial u(0, t)}{\partial x} = 0, \quad \frac{\partial u(L, t)}{\partial x} = 0$
Cauchy:   $u(0, t) = 0, \quad \frac{\partial u(L, t)}{\partial x} = 0$
Robin:   $u(0, t) = 1, \quad \frac{\partial u(L, t)}{\partial x} + u(L, t) = 0$
IBVP Example:   $$\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2}, \quad 0 \le x \le l, \, t > 0$$   $$\text{BCs: } u(0, t) = 0, \quad u(l, t) = 0 \quad (\text{Dirichlet})$$   $$\text{IC: } u(x, 0) = x$$
2. One-Dimensional Heat Flow & Diffusion Models
2.1 Derivation of the 1D Heat Equation
Consider a thin, homogeneous rod of length  $L$  lying along the x-axis, laterally insulated so heat flows strictly in the axial direction.
       Perfectly Insulated Lateral Surface
  +-------------------------------------------+
  |  |                 |x    x+dx|          | |
  |  |                 |<------->|          | |  ---> Heat Flow (x-axis)
  +-------------------------------------------+
  x=0                                         x=L


Parameters:
$A$ : Cross-sectional area of the rod
$\rho$ : Density of the material
$\sigma$ : Specific heat capacity
$k$ : Thermal conductivity
$u(x,t)$ : Temperature at position  $x$  and time  $t$
Energy Balance Equations:
Infinitesimal Element Mass and Heat Energy:  For an element  $x, x + \Delta x$ :  $$\Delta V = A \Delta x, \quad \Delta m = \rho A \Delta x$$  Heat contained at time  $t$ :  $$Q = \sigma \Delta m \, u(x, t) = \sigma \rho A u(x, t) \Delta x$$
Thermal Energy Change over time  $\Delta t$  :   $$\Delta Q = Q_1 - Q = \sigma \rho A u(x, t + \Delta t) - u(x, t) \Delta x$$
Fourier's Law of Heat Conduction:  Heat flux is proportional to the negative temperature gradient:  $$Q_{\text{flux}} = -k \frac{\partial u}{\partial x} = -k u_x(x, t)$$  Net heat entering at  $x$  minus heat leaving at  $x + \Delta x$  during  $\Delta t$ :  $$\Delta Q = -k u_x(x, t) - (-k u_x(x + \Delta x, t)) \Delta t = k u_x(x + \Delta x, t) - u_x(x, t) \Delta t$$
Governing PDE Formulation:  Equating the two expressions for  $\Delta Q$ :  $$\sigma \rho A u(x, t + \Delta t) - u(x, t) \Delta x = k u_x(x + \Delta x, t) - u_x(x, t) \Delta t$$   $$\frac{u(x, t + \Delta t) - u(x, t)}{\Delta t} = \left(\frac{k}{\sigma \rho A}\right) \frac{u_x(x + \Delta x, t) - u_x(x, t)}{\Delta x}$$  Taking limits  $\Delta x, \Delta t \to 0$  yields the  1D Heat (Diffusion) Equation :  $$\frac{\partial u}{\partial t} = c^2 \frac{\partial^2 u}{\partial x^2}, \quad \text{where } c^2 = \frac{k}{\sigma \rho A} \quad (\text{thermal diffusivity } D)$$
2.2 Method of Separation of Variables for the 1D Heat Equation
Assume a product solution:  $u(x, t) = X(x)T(t)$ .
Substituting into  $\frac{\partial u}{\partial t} = c^2 \frac{\partial^2 u}{\partial x^2}$ :  $$X(x)T'(t) = c^2 X''(x)T(t) \implies \frac{X''(x)}{X(x)} = \frac{1}{c^2} \frac{T'(t)}{T(t)} = \mu \quad (\text{separation constant})$$
This creates two ordinary differential equations:  $$X''(x) - \mu X(x) = 0 \quad \text{and} \quad T'(t) - \mu c^2 T(t) = 0$$
Solution Cases for Separation Constant  $\mu$ :
Case I: Positive Separation Constant (  $\mu = \lambda^2 > 0$  )   $$X(x) = A_1 \cosh(\lambda x) + A_2 \sinh(\lambda x), \quad T(t) = A_3 e^{\lambda^2 c^2 t}$$   $$u(x, t) = C_1 \cosh(\lambda x) + C_2 \sinh(\lambda x) e^{\lambda^2 c^2 t}$$   Physical Interpretation:  As  $t \to \infty$ ,  $u(x, t) \to \infty$ . This exponential growth is physically unrealistic for passive heat conduction and is rejected for standard bounded problems.
Case II: Zero Separation Constant (  $\mu = 0$  )   $$X(x) = A_4 x + A_5, \quad T(t) = A_6$$   $$u(x, t) = C_3 x + C_4$$   Physical Interpretation:  Represents steady-state linear temperature distribution.
Case III: Negative Separation Constant (  $\mu = -\lambda^2 < 0$  )   $$X(x) = A_7 \cos(\lambda x) + A_8 \sin(\lambda x), \quad T(t) = A_9 e^{-\lambda^2 c^2 t}$$   $$u(x, t) = C_5 \cos(\lambda x) + C_6 \sin(\lambda x) e^{-\lambda^2 c^2 t}$$   Physical Interpretation:  Models exponentially decaying spatial temperature waves over time, consistent with real thermal relaxation.
General Solution Structure Criteria
Homogeneous Dirichlet BCs (  $u(0, t) = 0, u(L, t) = 0$  ):  Case III ( $\mu = -\lambda^2$ ) provides the only non-trivial solution.
Non-homogeneous Dirichlet BCs (  $u(0, t) = \alpha, u(L, t) = \beta$  ):  Formed by a combination of Case II (steady-state) and Case III (transient decay).
Neumann / Robin BCs:  Formed by combinations of Case II and Case III solutions.
2.3 Solved Examples: 1D Heat Equation
Example 2.1: Uniform Rod with Cool Ends (Dirichlet BCs)
A rod of length  $20\text{ cm}$  ( $D = 0.475$ ) with insulated sides has its ends suddenly cooled to  $0^\circ\text{C}$  and maintained at  $0^\circ\text{C}$ . Initial temperature distribution:  $$u(x, 0) = \begin{cases} x, & 0 \le x \le 10 \\ 20 - x, & 10 \le x \le 20 \end{cases}$$
Model Formulation:   $$\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2}, \quad 0 \le x \le 20, \, t > 0$$   $$\text{BCs: } u(0, t) = 0, \quad u(20, t) = 0$$   $$\text{IC: } u(x, 0) = f(x)$$
Separation of Variables & Spatial Eigenvalues:  Setting  $\mu = -\lambda^2$ :  $$X(0) = 0 \implies C_1 = 0$$   $$X(20) = 0 \implies C_2 \sin(20\lambda) = 0 \implies \lambda_n = \frac{n\pi}{20}, \quad n = 1, 2, 3, \dots$$
Superposition and Fourier Coefficients:   $$u(x, t) = \sum_{n=1}^{\infty} B_n \sin\left(\frac{n\pi x}{20}\right) e^{-\frac{n^2 \pi^2 D}{400} t}$$  Applying the initial condition via half-range Fourier sine series:  $$B_n = \frac{2}{20} \left\int_{0}^{10} x \sin\left(\frac{n\pi x}{20}\right) dx + \int_{10}^{20} (20 - x) \sin\left(\frac{n\pi x}{20}\right) dx \right = \frac{80}{n^2 \pi^2} \sin\left(\frac{n\pi}{2}\right)$$   $$B_n = \begin{cases} 0, & n = 2m \text{ (even)} \\ \frac{80}{(2m - 1)^2 \pi^2}, & n = 2m - 1 \text{ (odd)} \end{cases}$$
Final Solution:   $$u(x, t) = \sum_{m=1}^{\infty} \frac{80}{(2m - 1)^2 \pi^2} \sin\left(\frac{(2m - 1)\pi x}{20}\right) e^{-\frac{(2m - 1)^2 \pi^2 D}{400} t}$$
Example 2.2: Rod with Insulated Ends (Neumann BCs)
A rod of length  $100\text{ cm}$  ( $D = 0.475$ ) has both ends insulated:  $\frac{\partial u(0, t)}{\partial x} = 0, \frac{\partial u(100, t)}{\partial x} = 0$ . Initial temperature:  $$u(x, 0) = \begin{cases} x, & 0 \le x \le 40 \\ 100 - x, & 40 \le x \le 100 \end{cases}$$
Model Formulation:   $$\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2}, \quad 0 \le x \le 100, \, t > 0$$   $$\text{BCs: } u_x(0, t) = 0, \quad u_x(100, t) = 0$$
Case Analysis for  $\mu$  :
$\mu > 0 \implies$  Trivial solution.
$\mu = 0 \implies X(x) = \text{constant} \implies u_0(x, t) = B_0$ .
$\mu = -\lambda^2 < 0 \implies X'(x) = -C_1 \lambda \sin(\lambda x) + C_2 \lambda \cos(\lambda x)$ .  $$X'(0) = 0 \implies C_2 = 0; \quad X'(100) = 0 \implies \sin(100\lambda) = 0 \implies \lambda_n = \frac{n\pi}{100}$$
Complete Solution Structure:   $$u(x, t) = B_0 + \sum_{n=1}^{\infty} B_n \cos\left(\frac{n\pi x}{100}\right) e^{-\frac{n^2 \pi^2 D}{10000} t}$$
Fourier Coefficients:   $$B_0 = \frac{1}{100} \left\int_{0}^{40} x dx + \int_{40}^{100} (100 - x) dx \right = 25$$   $$B_n = \frac{2}{100} \int_{0}^{100} u(x, 0) \cos\left(\frac{n\pi x}{100}\right) dx = \frac{200}{n^2 \pi^2} \left2\cos\left(\frac{n\pi}{2}\right) - \cos(n\pi) - 1 \right$$
Final Analytical Solution:   $$u(x, t) = 25 + \sum_{n=1}^{\infty} \frac{200}{n^2 \pi^2} \left2\cos\left(\frac{n\pi}{2}\right) - \cos(n\pi) - 1 \right \cos\left(\frac{n\pi x}{100}\right) e^{-\frac{n^2 \pi^2 D}{10000} t}$$
3. Two-Dimensional Heat Equation
3.1 Model Formulation
For a thin rectangular plate of dimensions  $a \times b$ , insulated on top and bottom faces with zero edge temperatures:
$$\frac{\partial u}{\partial t} = k \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right), \quad 0 \le x \le a, \, 0 \le y \le b$$   $$\text{BCs: } u(0, y, t) = u(a, y, t) = 0; \quad u(x, 0, t) = u(x, b, t) = 0$$   $$\text{IC: } u(x, y, 0) = f(x, y)$$
3.2 Solution via Separation of Variables
Substitute  $u(x, y, t) = X(x)Y(y)T(t)$ :  $$\frac{X''}{X} + \frac{Y''}{Y} = \frac{T'}{kT}$$
Because  $x, y, t$  are independent variables, each ratio equals a separate separation constant:  $$\frac{X''}{X} = \mu_1 = -\lambda_1^2, \quad \frac{Y''}{Y} = \mu_2 = -\lambda_2^2, \quad \frac{T'}{kT} = \mu_1 + \mu_2 = -(\lambda_1^2 + \lambda_2^2)$$
Solving the spatial ODEs with boundary conditions yields:  $$X_m(x) = \sin\left(\frac{m\pi x}{a}\right), \quad \lambda_1 = \frac{m\pi}{a}, \quad m = 1, 2, 3, \dots$$   $$Y_n(y) = \sin\left(\frac{n\pi y}{b}\right), \quad \lambda_2 = \frac{n\pi}{b}, \quad n = 1, 2, 3, \dots$$   $$T_{mn}(t) = C e^{-\lambda_{mn}^2 t}, \quad \text{where } \lambda_{mn}^2 = k \pi^2 \left( \frac{m^2}{a^2} + \frac{n^2}{b^2} \right)$$
General Superposition Solution:
$$u(x, y, t) = \sum_{n=1}^{\infty} \sum_{m=1}^{\infty} A_{mn} \sin\left(\frac{m\pi x}{a}\right) \sin\left(\frac{n\pi y}{b}\right) e^{-\lambda_{mn}^2 t}$$
The coefficients  $A_{mn}$  are derived using the double Fourier sine series expansion of  $f(x, y)$ :  $$A_{mn} = \frac{4}{ab} \int_{x=0}^{a} \int_{y=0}^{b} f(x, y) \sin\left(\frac{m\pi x}{a}\right) \sin\left(\frac{n\pi y}{b}\right) dy \, dx$$
3.3 Example 3.1: Polynomial Initial Temperature Distribution
For initial distribution  $f(x, y) = x y (\pi - x) (\pi - y)$ :  $$A_{mn} = \frac{4}{ab} \left\int_{0}^{a} (\pi x - x^2) \sin\left(\frac{m\pi x}{a}\right) dx \right \times \left\int_{0}^{b} (\pi y - y^2) \sin\left(\frac{n\pi y}{b}\right) dy \right$$   $$A_{mn} = \frac{16 a^2 b^2}{m^3 n^3 \pi^6} 1 - (-1)^m = \begin{cases} 0, & m \text{ or } n \text{ is even} \\ \frac{64 a^2 b^2}{m^3 n^3 \pi^6}, & m = 2p-1, n = 2q-1 \text{ (both odd)} \end{cases}$$
$$\text{Final Solution: } u(x, y, t) = \sum_{p=1}^{\infty} \sum_{q=1}^{\infty} \frac{64 a^2 b^2}{\pi^6 (2p-1)^3 (2q-1)^3} \sin\left(\frac{(2p-1)\pi x}{a}\right) \sin\left(\frac{(2q-1)\pi y}{b}\right) e^{-k\pi^2 \left(\frac{(2p-1)^2}{a^2} + \frac{(2q-1)^2}{b^2}\right) t}$$
4. Steady Heat Flow: Laplace Equation
When thermal equilibrium is reached, temperature becomes independent of time ( $\frac{\partial u}{\partial t} = 0$ ).
1D Steady State:   $\frac{\partial^2 u}{\partial x^2} = 0 \implies u(x) = C_1 x + C_2$  (Linear solutions).
2D Steady State:   $\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0 \implies \nabla^2 u = 0$  ( Laplace Equation ). Solutions to the Laplace equation are called  harmonic functions .
4.1 Dirichlet Problem on a Rectangular Domain
Find  $u(x, y)$  satisfying  $\nabla^2 u = 0$  given:  $$u(0, y) = 0, \quad u(a, y) = 0, \quad 0 \le y \le b$$   $$u(x, 0) = 0, \quad u(x, b) = f(x), \quad 0 \le x \le a$$
                   u(x,b) = f(x)
           +---------------------------+
           |                           |
u(0,y) = 0 |      ∇²u = 0              | u(a,y) = 0
           |                           |
           +---------------------------+
                   u(x,0) = 0


Separation of Variables:   $u(x, y) = X(x)Y(y) \implies \frac{X''}{X} = -\frac{Y''}{Y} = -\lambda^2$ .
Spatial Solutions:
$X(x) = A_2 \sin\left(\frac{n\pi x}{a}\right), \quad \lambda_n = \frac{n\pi}{a}$
$Y'' - \lambda_n^2 Y = 0 \implies Y(y) = B_1 e^{\frac{n\pi y}{a}} + B_2 e^{-\frac{n\pi y}{a}}$
$Y(0) = 0 \implies B_1 = -B_2 \implies Y(y) = 2 B_1 \sinh\left(\frac{n\pi y}{a}\right)$
Superposition Solution:   $$u(x, y) = \sum_{n=1}^{\infty} A_n \sin\left(\frac{n\pi x}{a}\right) \sinh\left(\frac{n\pi y}{a}\right)$$
Non-Homogeneous Top Boundary Evaluation:   $$u(x, b) = f(x) = \sum_{n=1}^{\infty} \leftA_n \sinh\left(\frac{n\pi b}{a}\right) \right \sin\left(\frac{n\pi x}{a}\right)$$   $$A_n = \frac{2}{a \sinh\left(\frac{n\pi b}{a}\right)} \int_{0}^{a} f(x) \sin\left(\frac{n\pi x}{a}\right) dx$$
Example 4.1 Solution ( $a=\pi, b=\pi, f(x) = x(\pi - x)$ ):
$$A_n = \frac{2}{\pi \sinh(n\pi)} \int_{0}^{\pi} x(\pi - x) \sin(n x) dx = \frac{4}{\pi n^3 \sinh(n\pi)} 1 - (-1)^n$$   $$u(x, y) = \sum_{m=1}^{\infty} \frac{8}{\pi (2m-1)^3 \sinh((2m-1)\pi)} \sin((2m-1)x) \sinh((2m-1)y)$$
4.2 Neumann Problem on a Rectangular Domain
For specified normal derivatives across boundaries:  $$u_x(0, y) = 0, \quad u_x(a, y) = 0, \quad 0 \le y \le b$$   $$u_y(x, 0) = 0, \quad u_y(x, b) = f(x), \quad 0 \le x \le a$$
Spatial Solutions:   $$X(x) = A_1 \cos\left(\frac{n\pi x}{a}\right), \quad Y(y) = 2 B_1 \cosh\left(\frac{n\pi y}{a}\right)$$
Superposition Solution:   $$u(x, y) = \sum_{n=1}^{\infty} A_n \cos\left(\frac{n\pi x}{a}\right) \cosh\left(\frac{n\pi y}{a}\right)$$
Top Derivative Boundary Condition (  $u_y(x, b) = f(x)$  ):   $$f(x) = \sum_{n=1}^{\infty} A_n \left(\frac{n\pi}{a}\right) \sinh\left(\frac{n\pi b}{a}\right) \cos\left(\frac{n\pi x}{a}\right)$$   $$A_n = \frac{2}{n\pi \sinh\left(\frac{n\pi b}{a}\right)} \int_{0}^{a} f(x) \cos\left(\frac{n\pi x}{a}\right) dx$$
5. Wave Equations & Oscillatory Systems
5.1 Physical Derivation of the 1D Wave Equation
Consider a thin, flexible string of length  $L$  and linear mass density  $\rho$  stretched under constant high tension  $T$ .
                 T₂
                ^  /  Q (x₂, u(x₂,t))
               /  /
              /  / 
    P        /  /  s+ds
   (x₁,u(x₁,t))  
    \  /
     \/
     T₁


Modeling Assumptions:
Negligible bending resistance and negligible gravity relative to tension ( $T \gg \text{gravity}$ ).
Motion is strictly lateral (vertical in the  $y$ -plane).
Force Balance Equations:
Horizontal Force Equilibrium:   $$T_2 \cos\beta - T_1 \cos\alpha = 0 \implies T_1 \cos\alpha = T_2 \cos\beta = T \quad (\text{constant})$$
Vertical Motion Equation (Newton's Second Law):   $$(\rho \Delta s) \frac{\partial^2 u}{\partial t^2} = T_2 \sin\beta - T_1 \sin\alpha$$
Division by  $T$  :   $$\frac{\rho \Delta s}{T} \frac{\partial^2 u}{\partial t^2} = \frac{T_2 \sin\beta}{T_2 \cos\beta} - \frac{T_1 \sin\alpha}{T_1 \cos\alpha} = \tan\beta - \tan\alpha = u_x(x_2, t) - u_x(x_1, t)$$
Limit as  $\Delta x \to 0$  (  $\Delta s \to \Delta x$  ):   $$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}, \quad \text{where } c^2 = \frac{T}{\rho} \quad (c = \text{propagation speed})$$
5.2 Vibrating String IBVP Solution
$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}, \quad 0 \le x \le L, \, t > 0$$   $$\text{BCs: } u(0, t) = 0, \quad u(L, t) = 0$$   $$\text{ICs: } u(x, 0) = f_1(x) \quad (\text{initial displacement}), \quad \frac{\partial u(x, 0)}{\partial t} = f_2(x) \quad (\text{initial velocity})$$
Solution Steps:
Separation of Variables:   $u(x, t) = X(x)T(t) \implies \frac{X''}{X} = \frac{T''}{c^2 T} = -\lambda^2$ .
Eigenvalues and Temporal Solutions:   $$\lambda_n = \frac{n\pi}{L}, \quad X_n(x) = \sin\left(\frac{n\pi x}{L}\right)$$   $$T_n(t) = A_n \cos\left(\frac{cn\pi t}{L}\right) + B_n \sin\left(\frac{cn\pi t}{L}\right)$$
General Series Solution:   $$u(x, t) = \sum_{n=1}^{\infty} \leftA_n \cos\left(\frac{cn\pi t}{L}\right) + B_n \sin\left(\frac{cn\pi t}{L}\right) \right \sin\left(\frac{n\pi x}{L}\right)$$
Fourier Coefficients:   $$A_n = \frac{2}{L} \int_{0}^{L} f_1(x) \sin\left(\frac{n\pi x}{L}\right) dx$$   $$B_n = \frac{2}{n\pi c} \int_{0}^{L} f_2(x) \sin\left(\frac{n\pi x}{L}\right) dx$$
Important Corollaries:
Corollary 1 (Released from Rest,  $f_2(x) = 0$  ):   $$B_n = 0 \implies u(x, t) = \sum_{n=1}^{\infty} A_n \cos\left(\frac{cn\pi t}{L}\right) \sin\left(\frac{n\pi x}{L}\right)$$
Corollary 2 (Zero Initial Deflection,  $f_1(x) = 0$  ):   $$A_n = 0 \implies u(x, t) = \sum_{n=1}^{\infty} B_n \sin\left(\frac{cn\pi t}{L}\right) \sin\left(\frac{n\pi x}{L}\right)$$
5.3 Example 5.1: Plucked Guitar String
A string fixed at  $(0,0)$  and  $(L,0)$  is plucked from rest with initial position  $u(x,0) = \mu x(L - x)$ .
Initial Conditions:   $f_1(x) = \mu x(L - x)$ ,  $f_2(x) = 0 \implies B_n = 0$ .
Coefficient Calculation:   $$A_n = \frac{2}{L} \int_{0}^{L} \mu x (L - x) \sin\left(\frac{n\pi x}{L}\right) dx = \frac{2\mu}{L} \left2 \left(\frac{L}{n\pi}\right)^3 \{1 - (-1)^n\} \right$$   $$A_n = \begin{cases} \frac{8\mu L^2}{n^3 \pi^3}, & n \text{ is odd} \\ 0, & n \text{ is even} \end{cases}$$
Displacement Function:   $$u(x, t) = \sum_{n=1}^{\infty} \frac{8\mu L^2}{(2n-1)^3 \pi^3} \cos\left( \frac{(2n-1)c\pi t}{L} \right) \sin\left( \frac{(2n-1)\pi x}{L} \right)$$
6. Two-Dimensional Wave Equation
6.1 Model Formulation
For a thin elastic membrane stretched over a rectangular frame  $a \times b$  with fixed edges:
$$\frac{\partial^2 u}{\partial t^2} = c^2 \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right), \quad 0 \le x \le a, \, 0 \le y \le b$$   $$\text{BCs: } u(0, y, t) = u(a, y, t) = 0; \quad u(x, 0, t) = u(x, b, t) = 0$$   $$\text{ICs: } u(x, y, 0) = f(x, y), \quad \frac{\partial u(x, y, 0)}{\partial t} = g(x, y)$$
6.2 Separation of Variables Analysis
Substitute  $u(x, y, t) = X(x)Y(y)T(t)$ :  $$\frac{X''}{X} + \frac{Y''}{Y} = \frac{T''}{c^2 T}$$
Assigning separation constants:  $$\frac{X''}{X} = -\lambda_1^2 \implies X_m(x) = \sin\left(\frac{m\pi x}{a}\right), \quad \lambda_1 = \frac{m\pi}{a}$$   $$\frac{Y''}{Y} = -\lambda_2^2 \implies Y_n(y) = \sin\left(\frac{n\pi y}{b}\right), \quad \lambda_2 = \frac{n\pi}{b}$$   $$T'' + \lambda_{mn}^2 T = 0, \quad \text{where } \lambda_{mn}^2 = c^2 \pi^2 \left( \frac{m^2}{a^2} + \frac{n^2}{b^2} \right)$$
Temporal Solution Component:
$$T_{mn}(t) = A_{mn} \cos(\lambda_{mn} t) + B_{mn} \sin(\lambda_{mn} t)$$
Double Series Representation:
$$u(x, y, t) = \sum_{n=1}^{\infty} \sum_{m=1}^{\infty} \leftA_{mn} \cos(\lambda_{mn} t) + B_{mn} \sin(\lambda_{mn} t) \right \sin\left(\frac{m\pi x}{a}\right) \sin\left(\frac{n\pi y}{b}\right)$$
Where  $A_{mn}$  and  $B_{mn}$  are determined from initial displacement  $f(x, y)$  and velocity  $g(x, y)$  via double Fourier sine expansions:  $$A_{mn} = \frac{4}{ab} \int_{x=0}^{a} \int_{y=0}^{b} f(x, y) \sin\left(\frac{m\pi x}{a}\right) \sin\left(\frac{n\pi y}{b}\right) dy \, dx$$   $$B_{mn} = \frac{4}{ab \, \lambda_{mn}} \int_{x=0}^{a} \int_{y=0}^{b} g(x, y) \sin\left(\frac{m\pi x}{a}\right) \sin\left(\frac{n\pi y}{b}\right) dy \, dx$$
7. Comparative Summary of Fundamental Spatial PDEs
System Model,PDE Equation,Spatial Eigenfunctions,Natural Frequencies / Decay Factors
1D Heat Flow,$\frac{\partial u}{\partial t} = c^2 \frac{\partial^2 u}{\partial x^2}$,$\sin\left(\frac{n\pi x}{L}\right)$  or  $\cos\left(\frac{n\pi x}{L}\right)$,Exponential Decay:  $e^{-\frac{n^2 \pi^2 c^2}{L^2} t}$
2D Heat Flow,$\frac{\partial u}{\partial t} = k \nabla^2 u$,$\sin\left(\frac{m\pi x}{a}\right) \sin\left(\frac{n\pi y}{b}\right)$,Exponential Decay:  $e^{-k \pi^2 \left(\frac{m^2}{a^2} + \frac{n^2}{b^2}\right) t}$
2D Steady Heat,$\nabla^2 u = 0$,$\sin\left(\frac{n\pi x}{a}\right) \sinh\left(\frac{n\pi y}{a}\right)$,Spatial Growth/Decay along non-periodic axis
1D Wave (String),$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$,$\sin\left(\frac{n\pi x}{L}\right)$,"Harmonic Oscillation:  $\cos\left(\frac{cn\pi t}{L}\right), \sin\left(\frac{cn\pi t}{L}\right)$"
2D Wave (Membrane),$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$,$\sin\left(\frac{m\pi x}{a}\right) \sin\left(\frac{n\pi y}{b}\right)$,"Harmonic Oscillation:  $\cos(\lambda_{mn} t), \sin(\lambda_{mn} t)$"

