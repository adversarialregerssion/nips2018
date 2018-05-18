"""
Generates adversarial noise for autoencoders. Supports grayscale and multi channel
autoencoders. Colorization is also supported.
"""

import numpy as np
import tensorflow as tf
import scipy
from time import clock
import random

import datasets
import models
import parser
import helpers
import time

tf.reset_default_graph()

def adv_noise(value, params, epsilon_range, method="m2", norm="l2", preload_gradient=False, num_iterations=10):

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Graph().as_default(), tf.Session(config = config) as sess:

        if preload_gradient == True:
            # Import checkpoint
            saver = tf.train.import_meta_graph("{}/{}/grad/grad.ckpt.meta".format(
                params["models_dir"], params["model"] ))
            saver.restore(sess, tf.train.latest_checkpoint("{}/{}/grad".format(
                params["models_dir"], params["model"]
            )))
        else:
            # Import existing model
            saver = tf.train.import_meta_graph("{}/{}/{}/{}".format(
                params["models_dir"], params["model"], params["dataset"], params["graph_file"] ))
            saver.restore(sess, tf.train.latest_checkpoint("{}/{}/{}".format(
                params["models_dir"], params["model"], params["dataset"] )))

        graph = tf.get_default_graph()

        # print( [n.name for n in tf.get_default_graph().as_graph_def().node] )

        prod = np.prod(params["image_dims"])
        input = graph.get_tensor_by_name(params["input"])
        output = graph.get_tensor_by_name(params["output"])

        _psnr = []
        _images = []

        if method == "quadratic":

            if params["colors_input"] == "y":
                prod = np.prod(np.array([params["image_dims"][0], params["image_dims"][1], 1]))

            # Calculate jacobian
            if preload_gradient == True:
                grad = graph.get_tensor_by_name("jacobian:0")
            else:
                _grad = []

                ### flatten output
                _output = tf.reshape(output, [prod, 1])

                for x in range(prod):
                    _grad.append( tf.gradients(tf.gather(_output, x, axis=0), input)[0] )
                grad = tf.stack(_grad, name="jacobian")

                # Save gradient checkpoint
                saver = tf.train.Saver()
                save_path = saver.save(sess, "./{}/{}/grad/grad.ckpt".format(
                    params["models_dir"], params["model"]
                ))

            for epsilon in epsilon_range:

                mse = []
                images = []

                for v in value:

                    if params["colors_input"] == "y":
                        v = v[:,:,0]
                        v = np.reshape(v, [params["image_dims"][0], params["image_dims"][1], 1])

                    fY = sess.run([output], feed_dict={input: np.array([v])})
                    w = None
                    x = v.copy()

                    eta_t = np.zeros(x.shape)
                    used_pixels = []
                    for iter in range(num_iterations):
                        t = time.time()

                        if norm == "l2":
                            dX = sess.run([grad], feed_dict={input: np.array([x])})

                            # Eigenvector calculation
                            dX = np.reshape(dX, (prod, prod))
                            _dX = np.dot(dX.T, dX)

                            ev = np.array(scipy.linalg.eigh(_dX, eigvals=(prod-1, prod-1))[1]).T

                            # Calculate fooledY
                            _fX = np.reshape(np.array([x]), (1, prod))
                            w = _fX + ((epsilon/num_iterations) * ev)
                            x = np.array(w).reshape(np.array(x).shape)

                            # fooledY = sess.run([output], feed_dict={input: np.array([x])})

                        elif norm == "linf":

                            # print("EMILIO QUADRATIC LINF SOLUTION")
                            dX = sess.run([grad], feed_dict={input: np.array([x])})
                            dX = np.array(dX)
                            dX = np.reshape(dX, np.concatenate((np.array([prod]), params["image_dims"]), axis=0))
                            [K, H, W, D] = dX.shape
                            norms = np.zeros([H,W,D])
                            idx_mtx = np.zeros([3, H,W,D])
                            for hh in range(H):
                                for ww in range(W):
                                    for dd in range(D):
                                        norms[hh,ww,dd] = np.sum(dX[:,hh,ww,dd].flatten()**2)
                                        idx_mtx[:, hh, ww, dd] = np.array([hh, ww, dd])

                            idx = np.argsort(norms.flatten())[::-1]
                            Hvec = idx_mtx[0,:,:,:].flatten().astype('int32')
                            Wvec = idx_mtx[1, :, :, :].flatten().astype('int32')
                            Dvec = idx_mtx[2, :, :, :].flatten().astype('int32')

                            rho = np.zeros(norms.shape)
                            rho[Hvec[idx[0]], Wvec[idx[0]], Dvec[idx[0]] ] = 1
                            Jvec = dX[:, Hvec[idx[0]], Wvec[idx[0]], Dvec[idx[0]]]
                            for kk in range(len(idx)-1):
                                Jk = dX[:, Hvec[idx[kk+1]], Wvec[idx[kk+1]], Dvec[idx[kk+1]]]
                                rho[Hvec[idx[kk+1]], Wvec[idx[kk+1]], Dvec[idx[kk+1]]] = np.sign(np.matmul(Jvec.T, Jk))
                                Jvec = Jvec + rho[Hvec[idx[kk+1]], Wvec[idx[kk+1]], Dvec[idx[kk+1]]]*Jk

                            if iter==0:
                                eta_t = eta_t + (epsilon / num_iterations) * rho
                            else:
                                eta_t = eta_t + np.sign(np.matmul(eta_t.T, rho))*(epsilon / num_iterations) * rho
                            x = v + eta_t
                        elif norm == "pixel":
                            dX = sess.run([grad], feed_dict={input: np.array([x])})
                            dX = np.array(dX)
                            dX = np.reshape(dX, np.concatenate((np.array([prod]), params["image_dims"]), axis=0))
                            [K, H, W, D] = dX.shape
                            norms = np.zeros([H,W,D])
                            idx_mtx = np.zeros([3, H,W,D])

                            Jvec_norm = 0
                            for hh in range(H):
                                for ww in range(W):
                                    for dd in range(D):
                                        norms[hh,ww,dd] = np.sum(dX[:,hh,ww,dd].flatten()**2)
                                        idx_mtx[:, hh, ww, dd] = np.array([hh, ww, dd])
                                    idx = np.argsort(norms[hh,ww,:])[::-1]
                                    rho = np.zeros(D)
                                    rho[idx[0]] = 1
                                    Jvec = dX[:,hh,ww,idx[0]]
                                    if len(idx)>1:
                                        for kk in range(len(idx) - 1):
                                            Jk = dX[:, hh, ww, idx[kk + 1]]
                                            rho[idx[kk + 1]] = np.sign(np.matmul(Jvec.T, Jk))
                                            Jvec = Jvec + rho[idx[kk + 1]] * Jk
                                    if (np.sum(Jvec.flatten()**2) >= Jvec_norm) and ([hh,ww] not in used_pixels):
                                        h_opt = hh
                                        w_opt = ww
                                        rho_opt = rho
                                        Jvec_norm = np.sum(Jvec.flatten()**2)

                            used_pixels.append([h_opt, w_opt])
                            eta = np.zeros([H, W, D])
                            eta[h_opt, w_opt, :] = epsilon*rho_opt
                            x = x + np.reshape(eta, x.shape)

                        print('Time per iteration = {} secs'.format(time.time() - t))


                    # Images
                    fooledY = sess.run([output], feed_dict={input: np.array([x])})

                    if params["colors_output"] == "cbcr":
                        X_org, X_new, fY, fooledY = helpers.merge_color_channels(params, v, x[0], fY, fooledY)
                    else:
                        # Images
                        X_org = np.array([[v]])
                        X_new = np.array([[x]])
                        fY = np.array(fY)
                        fooledY = np.array(fooledY)

                    images.append([X_org, X_new, fY, fooledY])

                    # Mean squared error
                    mse.append(helpers.mse(X_org, fooledY))

                    print(epsilon)

                avg_mse = np.mean(mse)
                psnr = helpers.psnr(avg_mse)
                _psnr.append(psnr)
                _images.append(images)

        elif method == "linear":

            for epsilon in epsilon_range:

                print("Current Epsilon {:.2f}".format(epsilon))

                mse = []
                images = []

                ii_imag = 0
                for x in value:

                    v = x.copy()

                    if params["colors_input"] == "y":
                        x = x[:,:,0]
                        x = np.reshape(x, [params["image_dims"][0], params["image_dims"][1], 1])
                        prod = np.prod(np.array([params["image_dims"][0], params["image_dims"][1], 1]))

                    x_org = x

                    # Change Loss Function
                    if params["description"] == "autoencoder":
                        true_Y = tf.placeholder(tf.float32, name="true_Y")
                        Y = 1.0
                        loss = tf.reduce_mean(tf.squared_difference(input, output))
                    else:
                        Y_ = np.array( sess.run([output], feed_dict={input: np.array([x_org])}) )
                        true_Y = tf.placeholder(tf.float32, name="true_Y", shape=Y_.shape)
                        sess.run(true_Y, feed_dict={true_Y: Y_})
                        loss = tf.reduce_mean(tf.squared_difference(true_Y, output))

                    grad = tf.gradients(loss, input)

                    used_pixels = []

                    t = time.time()
                    ii_imag += 1
                    for iter in range(num_iterations):
                        print('({}) {}-{}: Image {}/{} (iteration {}/{})'.format(str(epsilon),
                                                                                 method, norm, str(ii_imag),
                                                                                 str(len(value)), str(iter + 1),
                                                                                 num_iterations))

                        _x = np.reshape(x, prod)

                        fX = np.array(x).reshape(np.array([x_org]).shape)
                        dL = sess.run([grad], feed_dict={input: fX, true_Y: Y_})

                        dL = np.reshape(dL, fX.shape)
                        _dL = np.reshape(dL, prod)
                        dL_norm = helpers.l2_norm(dL)
                        _eta = np.zeros(prod)

                        if norm == "linf":
                            _eta = epsilon * np.sign(_dL)
                        elif norm == "l2":
                            _eta = epsilon * (_dL / dL_norm)
                        elif norm == "l1":
                            idx = np.where(np.abs(_dL) == np.abs(_dL).max())[0][0]
                            _eta[idx] = np.sign(_dL[idx]) * (epsilon / num_iterations)
                        elif norm == 'pixel':
                            [_, H, W, D] = dL.shape
                            norms = np.zeros([H, W])

                            tmp_norm = 0
                            h_opt = 0
                            w_opt = 0
                            n_opt = np.zeros(D)
                            for hh in range(H):
                                for ww  in range(W):
                                    norms[hh, ww] = np.sum(np.abs(dL[:,hh,ww,:]))
                                    if (norms[hh, ww] >= tmp_norm) and ([hh,ww] not in used_pixels):
                                        h_opt = hh
                                        w_opt = ww
                                        n_opt = epsilon*np.sign(dL[:,hh,ww,:])
                                        tmp_norm = norms[hh, ww]

                            used_pixels.append([h_opt, w_opt])
                            eta = np.zeros([H, W, D])
                            eta[h_opt, w_opt, :] = n_opt
                            _eta = eta.flatten()

                        _x = _x + _eta
                        x = np.reshape(_x, np.array([x_org]).shape)

                        # Only correct for first linearation
                        if iter == 0:
                            # Fix y == f(x) (gradient zero)
                            Y = np.array([[x_org]])
                            fY = np.array( sess.run([output], feed_dict={input: np.array([x_org])}) )

                            if np.array_equal(Y, fY):
                                print("Starting Point Initialization")
                                s = np.prod(fX.shape)
                                p = np.random.uniform(-epsilon/100000, epsilon/100000, size=s)
                                x = np.array(x) + np.reshape(p, np.array(x).shape)

                    print('Time per iteration = {} secs'.format(time.time() - t))

                    fY = sess.run([output], feed_dict={input: np.array([x_org]), true_Y: Y_})
                    fooledY = sess.run([output], feed_dict={input: np.array(x), true_Y: Y_})

                    if params["colors_output"] == "cbcr":
                        X_org, X_new, fY, fooledY = helpers.merge_color_channels(params, v, x[0], fY, fooledY)
                    else:
                        # Images
                        X_org = np.array([[x_org]])
                        X_new = np.array([x])
                        fY = np.array(fY)
                        fooledY = np.array(fooledY)

                    images.append([X_org, X_new, fY, fooledY])

                    # Mean squared error
                    mse.append(helpers.mse(X_org, fooledY))

                avg_mse = np.mean(mse)
                psnr = helpers.psnr(avg_mse)
                _psnr.append(psnr)
                _images.append(images)

        elif method == "rand":

            for epsilon in epsilon_range:

                mse = []
                images = []

                for v in value:

                    if params["colors_input"] == "y":
                        _v = v[:,:,0]
                        _v = np.reshape(_v, [params["image_dims"][0], params["image_dims"][1], 1])
                    else:
                        _v = v

                    # Predicted output
                    _fY = np.array( sess.run([output], feed_dict={input: np.array([_v])}) )

                    eta = np.zeros(len(_v))

                    if norm == "l2":
                        # Random noise
                        rnd = np.random.normal(size=_v.shape)
                        eta = epsilon * (rnd / helpers.l2_norm(rnd))
                    elif norm == "l1":
                        # Select index
                        shp = list(_v.shape)
                        idx = []
                        for i in range(len(shp)):
                            idx.append( random.randint(0, shp[i]-1) )
                        # Generate noise
                        eta = np.zeros(shp)
                        eta[idx] = epsilon * random.uniform(-1, 1)
                    elif norm == "linf":
                        # rnd = [random.uniform(-1, 1) for x in range(len(v))]
                        rnd = np.random.normal(size=_v.shape)
                        rnd = np.sign(rnd)
                        eta = epsilon * rnd
                    elif norm == 'pixel':
                        eta  = np.zeros(params["image_dims"])
                        ii=0
                        used_pixels = []
                        while ii < num_iterations:
                            hh = np.random.randint(0, params["image_dims"][0]-1)
                            ww = np.random.randint(0, params["image_dims"][1] - 1)
                            if [hh, ww] not in used_pixels:
                                eta[hh, ww, :] = epsilon*np.sign(np.random.normal(size=eta[hh, ww, :].shape))
                                ii += 1
                                used_pixels.append([hh, ww])
                        eta = np.reshape(eta, _v.shape)

                    _w = _v + eta
                    _fooledY = sess.run([output], feed_dict={input: np.array([_w])})

                    # Merge color channels
                    if params["colors_output"] == "cbcr":
                        X_org, X_new, fY, fooledY = helpers.merge_color_channels(params, v, _w, _fY, _fooledY)
                    else:
                        # Images
                        X_org = np.array([[_v]])
                        X_new = np.array([[_w]])
                        fY = np.array(_fY)
                        fooledY = np.array(_fooledY)

                    # Images
                    images.append([X_org, X_new, fY, fooledY])

                    # Mean squared error
                    mse.append(helpers.mse(np.array([[v]]), fooledY))

                avg_mse = np.mean(mse)
                psnr = helpers.psnr(avg_mse)
                _psnr.append(psnr)
                _images.append(images)

        return _psnr, _images

def main():
    params = parser.Parser().get_arguments()
    prod = np.prod(params["image_dims"])
    print(params)

    # Get datasets
    if params["dataset"] == "mnist":
        params["image_dims"] = [32, 32, 1]
        x_train, x_test = datasets.Dataset(params).mnist()
        x_train = x_train.reshape([-1, 32, 32, 1])
        x_test = x_test.reshape([-1, 32, 32, 1])
    elif params["dataset"] == "cifar":
        params["image_dims"] = [32, 32, 3]
        x_train, x_test = datasets.Dataset(params).cifar()
        x_train = x_train.reshape([-1, 32, 32, 3])
        x_test = x_test.reshape([-1, 32, 32, 3])
    elif params["dataset"] == "stl10":
        params["image_dims"] = [96, 96, 3]
        if params["colors_output"] == "rgb":
            x_test = datasets.Dataset(params).stl10(colors="rgb")
            x_test = x_test.reshape([-1, 96, 96, 3])
        elif params["colors_output"] == "cbcr":
            x_test = datasets.Dataset(params).stl10(colors="ycbr")
            x_test = x_test.reshape([-1, 96, 96, 3])

    # Import test image
    test_image = x_test[:10]

    # Get noise
    adv_dict = {
        'psnr_input': 0, 'rand-linf': 1, 'rand-l2': 2,
        'linear-linf-1': 3, 'linear-linf-10': 4, 'linear-linf-20': 5,
        'linear-l2-1': 6, 'linear-l2-10': 7, 'linear-l2-20': 8,
        'quadratic-linf-1': 9, 'quadratic-linf-10': 10,
        'quadratic-l2-1': 11, 'quadratic-l2-10': 12,
        'linear-pixel-1':13, 'linear-pixel-10':14,
        'quadratic-pixel-1':15,
        'rand-pixel-1': 16, 'rand-pixel-10': 17
    }
    img_mtx = {}

    if params["norm"] == "l2":
        # Epsilon
        epsilon_range = np.array(np.logspace(-0.5, 1.0, 5)) # for l2
        adv_mtx = np.zeros([len(epsilon_range), len(adv_dict)])

        # L2
        adv_mtx[:, adv_dict['psnr_input']] = -20.*np.log10(epsilon_range/np.sqrt(prod)) # only true for l2
        adv_mtx[:, adv_dict['rand-l2']], img_mtx['rand-l2'] = \
            adv_noise(test_image, params, epsilon_range, "rand", "l2", False)
        adv_mtx[:, adv_dict['linear-l2-1']], img_mtx['linear-l2-1'] = \
            adv_noise(test_image, params, epsilon_range, "linear", "l2", False, 1)
        adv_mtx[:, adv_dict['linear-l2-10']], img_mtx['linear-l2-10'] = \
            adv_noise(test_image, params, epsilon_range, "linear", "l2", False, 10)
        adv_mtx[:, adv_dict['linear-l2-20']], img_mtx['linear-l2-20'] = \
            adv_noise(test_image, params, epsilon_range, "linear", "l2", False, 20)
        if params['dataset'] == 'mnist' or params['dataset'] == 'cifar':
            adv_mtx[:, adv_dict['quadratic-l2-1']], img_mtx['quadratic-l2-1'] = \
                adv_noise(test_image, params, epsilon_range, "quadratic", "l2", False, 1)
            # adv_mtx[:, adv_dict['quadratic-l2-10']], img_mtx['quadratic-l2-10'] = \
            #     adv_noise(test_image, params, epsilon_range, "quadratic", "l2", True, 10)
            # helpers.save_image_fig(img_mtx, params, adv_mtx[:, adv_dict['psnr_input']],
            #     ["rand-l2", 'linear-l2-1', 'linear-l2-10', 'linear-l2-20', 'quadratic-l2-1', 'quadratic-l2-10'], "l2",
            #                          img_size=params["image_dims"])
        else:
            print('No image plots for l2')
            # helpers.save_image_fig(img_mtx, params, adv_mtx[:, adv_dict['psnr_input']],
            #                          ["rand-l2", 'linear-l2-1', 'linear-l2-10', 'linear-l2-20'],
            #                          "l2", img_size=params["image_dims"])

    elif params["norm"] == "linf":
        # Epsilon
        epsilon_range = np.array(np.logspace(-2, -0.5, 5)) # for linf
        adv_mtx = np.zeros([len(epsilon_range), len(adv_dict)])

        # Linf
        adv_mtx[:, adv_dict['psnr_input']] = -20. * np.log10(epsilon_range)
        adv_mtx[:, adv_dict['rand-linf']], img_mtx['rand-linf'] = \
            adv_noise(test_image, params, epsilon_range, "rand", "linf", False)
        adv_mtx[:, adv_dict['linear-linf-1']], img_mtx['linear-linf-1'] = \
            adv_noise(test_image, params, epsilon_range, "linear", "linf", False, 1)
        adv_mtx[:, adv_dict['linear-linf-10']], img_mtx['linear-linf-10'] = \
            adv_noise(test_image, params, epsilon_range, "linear", "linf", False, 10)
        adv_mtx[:, adv_dict['linear-linf-20']], img_mtx['linear-linf-20'] = \
            adv_noise(test_image, params, epsilon_range, "linear", "linf", False, 20)


        helpers.save_image_fig(img_mtx, params, adv_mtx[:, adv_dict['psnr_input']],
            ["rand-linf", "linear-linf-1", "linear-linf-10", "linear-linf-20"], "linf", img_size=params["image_dims"])

    elif params["norm"] == "pixel":
        # Epsilon
        epsilon_range = epsilon_range = np.array([0.1, 0.2, 0.3, 0.5, 0.7])
        adv_mtx = np.zeros([len(epsilon_range), len(adv_dict)])

        adv_mtx[:, adv_dict['psnr_input']] = epsilon_range # only true for l2

        adv_mtx[:, adv_dict['rand-pixel-1']], img_mtx['rand-pixel-1'] = \
            adv_noise(test_image, params, epsilon_range, "rand", "pixel", False, 1)

        adv_mtx[:, adv_dict['linear-pixel-1']], img_mtx['linear-pixel-1'] = \
            adv_noise(test_image, params, epsilon_range, "linear", "pixel", False, 1)

        adv_mtx[:, adv_dict['rand-pixel-10']], img_mtx['rand-pixel-10'] = \
            adv_noise(test_image, params, epsilon_range, "rand", "pixel", False, 100)

        adv_mtx[:, adv_dict['linear-pixel-10']], img_mtx['linear-pixel-10'] = \
            adv_noise(test_image, params, epsilon_range, "linear", "pixel", False, 100)

        helpers.save_image_fig(img_mtx, params, adv_mtx[:, adv_dict['psnr_input']],
                ['linear-pixel-1', 'linear-pixel-10', 'quadratic-pixel-1'], "pixel", img_size=params["image_dims"])

    np.savetxt(params['output_dir'] + '/summary/' + params['model'] + '_psnr_summary_' + params['dataset'] + '.csv', adv_mtx, delimiter=";")
    helpers.save_psnr_fig(adv_mtx, './results/images/' + params['figs_dir'] + '/' + params['model'] +'_fig_' + params['dataset'] + '_' + params["norm"] + '.png',
        adv_dict, legend=True)


if __name__ == '__main__':
    main()
