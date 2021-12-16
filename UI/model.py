
#   ******  Model below   ******
min_max_scaler = sklearn.preprocessing.MinMaxScaler()
min_max_scaler_test_stock1 = sklearn.preprocessing.MinMaxScaler()
min_max_scaler_test_stock2 = sklearn.preprocessing.MinMaxScaler()

def normalize_data(df):
    df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1,1))
    return df

# function to create train, validation, test data given stock data and sequence length
def load_data(real_data,stock, seq_len,flag):
    data_raw = stock.values # convert to numpy array
    data = []
    data_actual=[]
    data_actual_raw = real_data.values
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index: index + seq_len])
        data_actual.append(data_actual_raw[index: index + seq_len])
    data = np.array(data);
    data_actual = np.array(data_actual)
    valid_set_size = int(np.round(valid_set_size_percentage/100*data.shape[0]));
    test_set_size = int(np.round(test_set_size_percentage/100*data.shape[0]));
    train_set_size = data.shape[0] - (valid_set_size + test_set_size);
    x_train = data[:train_set_size,:-1,:]
    y_train = data_actual[:train_set_size,-1,:]
    y_train = min_max_scaler.fit_transform(y_train.reshape(-1,1))
    # x_valid = data[train_set_size:train_set_size+valid_set_size,:-1,:]
    # y_valid = data_actual[train_set_size:train_set_size+valid_set_size,-1,:]
    # y_valid = min_max_scaler.fit_transform(y_valid.reshape(-1,1))
    x_test = data[train_set_size:,:-1,:]
    y_test = data_actual[train_set_size:,-1,:]
    if(flag==1):
        y_test = min_max_scaler_test_stock1.fit_transform(y_test.reshape(-1,1))
    else:
        y_test = min_max_scaler_test_stock2.fit_transform(y_test.reshape(-1,1))
    return [x_train, y_train, x_test, y_test]



def predictStock(stock, date):
    try:
        if (stock == 'GM'):
            # print("GM")
            global GM_dates
            global GM_X
            global GM_Y

            if (date in GM_dates):
                result = np.where(GM_dates == date)
                # print(result)
                pred = predict_stock(GM_X[result[0]], stock)
                gm_pred = min_max_scaler_test_stock2.inverse_transform(pred[-1])
                return (gm_pred[0], GM_Y[result])
        else:
            global TT_dates
            global TT_X
            global TT_Y
            if (date in TT_dates):
                result = np.where(TT_dates == date)
                pred = predict_stock(TT_X[result[0]], stock)
                toyota_pred = min_max_scaler_test_stock1.inverse_transform(pred[-1])
                return (toyota_pred[0], TT_Y[result])
    except:
        return -1


def predict_stock(X_ip, stock):
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph('/home/saksham/Documents/microblog/app/dataset/my_test_model5.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint('/home/saksham/Documents/microblog/app/dataset/checkpoint'))
            # The input has 2 dimensions: dimension 0 is reserved for the first term and dimension 1 is reverved for the second term
            input_dimensions = 1
            batch_size = 185
            time_size = 20
            # Arbitrary number for the size of the hidden state
            hidden_size = 256

            class GRU:
                def __init__(self, input_dimensions, hidden_size, dtype=tf.float64):
                    self.input_dimensions = input_dimensions
                    self.hidden_size = hidden_size

                    # Weights for input vectors of shape (input_dimensions, hidden_size)
                    self.Wr = graph.get_tensor_by_name('Wr:0')
                    self.Wz = graph.get_tensor_by_name('Wz:0')
                    self.Wh = graph.get_tensor_by_name('Wh:0')

                    # Weights for hidden vectors of shape (hidden_size, hidden_size)
                    self.Ur = graph.get_tensor_by_name('Ur:0')
                    self.Uz = graph.get_tensor_by_name('Uz:0')
                    self.Uh = graph.get_tensor_by_name('Uh:0')

                    # Biases for hidden vectors of shape (hidden_size,)
                    self.br = graph.get_tensor_by_name('br:0')
                    self.bz = graph.get_tensor_by_name('bz:0')
                    self.bh = graph.get_tensor_by_name('bh:0')

                    # Define the input layer placeholder
                    self.input_layer = tf.placeholder(dtype=tf.float64, shape=(None, None, input_dimensions),
                                                      name='input')

                    # Put the time-dimension upfront for the scan operator
                    self.x_t = tf.transpose(self.input_layer, [1, 0, 2], name='x_t')

                    # A little hack (to obtain the same shape as the input matrix) to define the initial hidden state h_0
                    self.h_0 = tf.matmul(self.x_t[0, :, :],
                                         tf.zeros(dtype=tf.float64, shape=(input_dimensions, hidden_size)), name='h_0')

                    # Perform the scan operator
                    self.h_t_transposed = tf.scan(self.forward_pass, self.x_t, initializer=self.h_0,
                                                  name='h_t_transposed')

                    # Transpose the result back
                    self.h_t = tf.transpose(self.h_t_transposed, [1, 0, 2], name='h_t')

                def forward_pass(self, h_tm1, x_t):
                    """Perform a forward pass.

                    Arguments
                    ---------
                    h_tm1: np.matrix
                        The hidden state at the previous timestep (h_{t-1}).
                    x_t: np.matrix
                        The input vector.
                    """
                    # Definitions of z_t and r_t
                    z_t = tf.sigmoid(tf.matmul(x_t, self.Wz) + tf.matmul(h_tm1, self.Uz) + self.bz)
                    r_t = tf.sigmoid(tf.matmul(x_t, self.Wr) + tf.matmul(h_tm1, self.Ur) + self.br)

                    # Definition of h~_t
                    h_proposal = tf.tanh(
                        tf.matmul(x_t, self.Wh) + tf.matmul(tf.multiply(r_t, h_tm1), self.Uh) + self.bh)

                    # Compute the next hidden state
                    h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)

                    return h_t

            # Create a new instance of the GRU model
            gru = GRU(input_dimensions, hidden_size)
            W_output = graph.get_tensor_by_name('W_output:0')
            b_output = graph.get_tensor_by_name('b_output:0')
            output = tf.map_fn(lambda h_t: tf.matmul(h_t, W_output) + b_output, gru.h_t)
            output = tf.transpose(output, [1, 0, 2])

            class GRU1:
                def __init__(self, input_dimensions, hidden_size, dtype=tf.float64):
                    self.input_dimensions = input_dimensions
                    self.hidden_size = hidden_size

                    # Weights for input vectors of shape (input_dimensions, hidden_size)
                    self.Wr1 = graph.get_tensor_by_name('Wr1:0')
                    self.Wz1 = graph.get_tensor_by_name('Wz1:0')
                    self.Wh1 = graph.get_tensor_by_name('Wh1:0')

                    # Weights for hidden vectors of shape (hidden_size, hidden_size)
                    self.Ur1 = graph.get_tensor_by_name('Ur1:0')
                    self.Uz1 = graph.get_tensor_by_name('Uz1:0')
                    self.Uh1 = graph.get_tensor_by_name('Uh1:0')

                    # Biases for hidden vectors of shape (hidden_size,)
                    self.br1 = graph.get_tensor_by_name('br1:0')
                    self.bz1 = graph.get_tensor_by_name('bz1:0')
                    self.bh1 = graph.get_tensor_by_name('bh1:0')

                    # Define the input layer placeholder
                    self.input_layer = tf.placeholder(dtype=tf.float64, shape=(None, None, input_dimensions),
                                                      name='input')

                    # Put the time-dimension upfront for the scan operator
                    self.x_t = tf.transpose(self.input_layer, [1, 0, 2], name='x_t')

                    # A little hack (to obtain the same shape as the input matrix) to define the initial hidden state h_0
                    self.h_0 = tf.matmul(self.x_t[0, :, :],
                                         tf.zeros(dtype=tf.float64, shape=(input_dimensions, hidden_size)), name='h_0')

                    # Perform the scan operator
                    self.h_t_transposed = tf.scan(self.forward_pass, self.x_t, initializer=self.h_0,
                                                  name='h_t_transposed')

                    # Transpose the result back
                    self.h_t = tf.transpose(self.h_t_transposed, [1, 0, 2], name='h_t')

                def forward_pass(self, h_tm1, x_t):
                    """Perform a forward pass.

                    Arguments
                    ---------
                    h_tm1: np.matrix
                        The hidden state at the previous timestep (h_{t-1}).
                    x_t: np.matrix
                        The input vector.
                    """
                    # Definitions of z_t and r_t
                    z_t = tf.sigmoid(tf.matmul(x_t, self.Wz1) + tf.matmul(h_tm1, self.Uz1) + self.bz1)
                    r_t = tf.sigmoid(tf.matmul(x_t, self.Wr1) + tf.matmul(h_tm1, self.Ur1) + self.br1)

                    # Definition of h~_t
                    h_proposal = tf.tanh(
                        tf.matmul(x_t, self.Wh1) + tf.matmul(tf.multiply(r_t, h_tm1), self.Uh1) + self.bh1)

                    # Compute the next hidden state
                    h_t = tf.multiply(1 - z_t, h_tm1) + tf.multiply(z_t, h_proposal)

                    return h_t

            # Create a new instance of the GRU model
            gru1 = GRU1(input_dimensions, hidden_size)
            W_output1 = graph.get_tensor_by_name('W_output1:0')
            b_output1 = graph.get_tensor_by_name('b_output1:0')
            output1 = tf.map_fn(lambda h_t: tf.matmul(h_t, W_output1) + b_output1, gru1.h_t)
            output1 = tf.transpose(output1, [1, 0, 2])
            if (stock == 'GM'):
                pred = sess.run([output1], feed_dict={gru1.input_layer: X_ip})
            else:
                pred = sess.run([output], feed_dict={gru.input_layer: X_ip})
            return (pred[-1])



df = pd.read_csv('/home/saksham/Documents/microblog/app/dataset/toyota_final.csv')
dfg = pd.read_csv('/home/saksham/Documents/microblog/app/dataset/gm_final.csv')
df1 = df[['Close']]
dfg1 = dfg[['Close']]
valid_set_size_percentage = 10
test_set_size_percentage = 10
df_stock_norm = df1.copy()
df_stock_norm = normalize_data(df_stock_norm)
df_stock_real = df1.copy()
# create train, test data
seq_len = 20 # choose sequence length
x_train, y_train, x_test, y_test = load_data(df_stock_real,df_stock_norm, seq_len,1)
dates = df['Date'].values
TT_dates = dates[19:-1]
TT_X    =  np.concatenate((x_train,x_test), axis=0)
TT_Y    = df['Close'].values
TT_Y    = TT_Y[19:-1]
df_stock_norm1 = dfg1.copy()
df_stock_norm1 = normalize_data(df_stock_norm1)
# create train, test data
seq_len = 20 # choose sequence length
df_stock_real1 = dfg1.copy()
x_train1, y_train1, x_test1, y_test1 = load_data(df_stock_real1,df_stock_norm1, seq_len,0)
dates1 = dfg['Date'].values
GM_dates = dates1[19:-1]
GM_X    =  np.concatenate((x_train1,x_test1), axis=0)
GM_Y    = dfg['Close'].values
GM_Y    = GM_Y[19:-1]
