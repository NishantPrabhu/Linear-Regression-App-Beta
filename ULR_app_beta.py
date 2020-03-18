
# Dependencies
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt 
import streamlit as st 

# Title and customary text
st.title("Univariate Linear Regression")
st.write("This app has been made primarily for solving problems in **Inverse Methods for Heat Transfer**. You can adjust the parameters based on the problem you are solving; the app will perform linear regression and provide you some parameters to see how well it fits your data (visual and statistical). Or, you can upload a CSV file containing your data with the specifications given below and let the app do the rest.")

data_input = st.radio('How will you provide the data?', \
                       ['Parameter Input', 'Data Upload'], 0)

# Take action based on data_input
if data_input == 'Parameter Input':

    st.write('## **Parameter Input**')
    st.write('This feature is here for experimentation and learning. It will estimate the parameters for the following linear equation, arising out of 1D steady-state conduction with no internal heat generation.')
    r'''
    $$
    \frac{\partial}{\partial x}\left(k\frac{\partial T}{\partial x}\right)=0\quad\Rightarrow\quad \boxed{T(x)=\alpha_{0}+\alpha_{1}x}
    $$
    '''
    st.write('')
    r'''Below, input the values of $\alpha_{0}$ and $\alpha_{1}$ that you want. Also, provide the upper and lower limits for $x$ and the number of sample points you want between them.'''

    alpha_0 = st.number_input('Alpha 0 (between 0 and 5000)', 0, 5000, 300)
    alpha_1 = st.number_input('Alpha 1 (between -1000 and -1)', -1000, -1, -20)
    x_llim = st.number_input('Left limit for x in meters (between 0 and 1000)', 0, 1000, 0)
    x_rlim = st.number_input('Right limit for x in meters (between 0 and 1000, greater than the left limit)', 0, 1000, 10)

    if x_llim >= x_rlim:
        raise Exception('Left x limit is not strictly less than right x limit.')

    n_samples = st.number_input('How many sample points do you want? (between 2 and 1000)', 2, 1000, 100)

    noise_scale = st.slider('Noise variance (mean = 0.0)', 0.0, 10.0, 5.0, 0.1)

    # Main algorithm
    @st.cache
    def generate_data(alpha_0, alpha_1, x_llim, x_rlim, n_samples):
        x = np.linspace(x_llim, x_rlim, n_samples).reshape((-1, 1))
        T_real = (alpha_0 + alpha_1*x).reshape((-1, 1))
        noise = np.random.normal(loc=0, scale=noise_scale, size=T_real.shape[0]).reshape((-1, 1))
        T = (T_real + noise).reshape((-1, 1))
        return x, T
    
    x, T = generate_data(alpha_0, alpha_1, x_llim, x_rlim, n_samples)

    # Perform regression and generate plot
    # Perform the matrix operations to get alpha_0 and alpha_1
    A = np.array([[len(x), sum(x)],
                  [sum(x), sum(x**2)]], dtype=np.float64)
    b = np.array([sum(T),
                  sum(T * x)], dtype=np.float64)

    fit_params = np.dot(np.linalg.inv(A), b)

    # Visualizing the fit line
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(x, T, color='red', marker='o', label='data')
    x_vals = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
    fit_vals = fit_params[0][0] + fit_params[1][0]*x_vals
    ax.plot(x_vals, fit_vals, color='black', linewidth=3, label='fit line')
    plt.title('Fit line')
    plt.xlabel('x (m)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.grid()

    # Goodness of fit parameters ==========================================
    # Make a dataframe with x, T_noisy and predicted_T
    predicted_T = fit_params[0][0] + fit_params[1][0] * x
    data = np.vstack((x.reshape((1, -1)), T.reshape((1, -1)), predicted_T.reshape((1, -1)))).T
    df = pd.DataFrame(data, columns=['x', 'T_measured', 'T_fit'])

    df['Mean_deviation'] = (df['T_measured'] - df['T_measured'].mean())**2
    df['Fit_deviation'] = (df['T_measured'] - df['T_fit'])**2

    # Calculate goodness of fit parameters
    S_t, S_r = df['Mean_deviation'].sum(), df['Fit_deviation'].sum()
    R_2 = (S_t - S_r)/S_t
    SE = np.sqrt(S_r/(len(x) - 2))

    st.write("## **Regression Results**")
    st.write('')
    # Write fit params to output
    st.write('Estimated parameter values')
    st.write(pd.DataFrame(
        np.array([['Alpha 0', fit_params[0][0]],
                  ['Alpha 1', fit_params[1][0]]]),
        columns = ['Description', 'Value']
    ))

    st.write('Goodness of fit parameters')
    st.write(pd.DataFrame(
        np.array([['Coefficient of determination', R_2],
                  ['Correlation coefficient', R_2**0.5],
                  ['Standard error', SE]]),
        columns = ['Description', 'Value']
    ))
    st.write('')

    # Display plot
    if st.checkbox('Show fit line', value=False):
        st.pyplot(fig)

    # Checkboxes whether to display parity plot and residuals table
    if st.checkbox('Show parity plot', value=False):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.scatter(predicted_T, T, c='red', alpha=0.6)
        lim = np.linspace(*ax.get_xlim())
        ax.plot(lim, lim, color='black')
        plt.xlabel('Predictions')
        plt.ylabel('Target')
        plt.title('Parity plot')
        plt.grid()
        st.pyplot(fig)

    if st.checkbox('Show residuals table', value=False):
        st.write(df)

elif data_input == 'Data Upload':
    st.write('## **Data Upload**')
    st.write("Upload your data as a CSV file with two columns named 'x' and 'target'. Please ensure that your data doesn't have any missing values. The app will perform regression and show you the results in a jiffy!")
    
    up_file = st.file_uploader("Upload file", type="csv")

    if up_file is not None:
        data = pd.read_csv(up_file)
        x = data['x'].values
        T = data['target'].values

        # Perform regression and generate plot
        # Perform the matrix operations to get alpha_0 and alpha_1
        A = np.array([[len(x), sum(x)],
                    [sum(x), sum(x**2)]], dtype=np.float64)
        b = np.array([sum(T),
                    sum(T * x)], dtype=np.float64)

        fit_params = np.dot(np.linalg.inv(A), b)

        # Visualizing the fit line
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.scatter(x, T, color='red', marker='o', label='data')
        x_vals = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 1000)
        fit_vals = fit_params[0] + fit_params[1]*x_vals
        ax.plot(x_vals, fit_vals, color='black', linewidth=3, label='fit line')
        plt.title('Fit line')
        plt.xlabel('x (m)')
        plt.ylabel('Temperature (K)')
        plt.legend()
        plt.grid()

        # Goodness of fit parameters ==========================================
        # Make a dataframe with x, T_noisy and predicted_T
        predicted_T = fit_params[0] + fit_params[1] * x
        data = np.vstack((x.reshape((1, -1)), T.reshape((1, -1)), predicted_T.reshape((1, -1)))).T
        df = pd.DataFrame(data, columns=['x', 'T_measured', 'T_fit'])

        df['Mean_deviation'] = (df['T_measured'] - df['T_measured'].mean())**2
        df['Fit_deviation'] = (df['T_measured'] - df['T_fit'])**2

        # Calculate goodness of fit parameters
        S_t, S_r = df['Mean_deviation'].sum(), df['Fit_deviation'].sum()
        R_2 = (S_t - S_r)/S_t
        SE = np.sqrt(S_r/(len(x) - 2))

        st.write("## **Regression Results**")
        st.write('')
        # Write fit params to output
        st.write('Estimated parameter values')
        st.write(pd.DataFrame(
            np.array([['Alpha 0', fit_params[0]],
                    ['Alpha 1', fit_params[1]]]),
            columns = ['Description', 'Value']
        ))

        st.write('Goodness of fit parameters')
        st.write(pd.DataFrame(
            np.array([['Coefficient of determination', R_2],
                    ['Correlation coefficient', R_2**0.5],
                    ['Standard error', SE]]),
            columns = ['Description', 'Value']
        ))
        st.write('')

        # Display plot
        if st.checkbox('Show fit line', value=False):
            st.pyplot(fig)

        # Checkboxes whether to display parity plot and residuals table
        if st.checkbox('Show parity plot', value=False):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            ax.scatter(predicted_T, T, c='red', alpha=0.6)
            lim = np.linspace(*ax.get_xlim())
            ax.plot(lim, lim, color='black')
            plt.xlabel('Predictions')
            plt.ylabel('Target')
            plt.title('Parity plot')
            plt.grid()
            st.pyplot(fig)

        if st.checkbox('Show residuals table', value=False):
            st.write(df)
    


