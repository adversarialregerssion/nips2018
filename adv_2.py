"""
Generates adversarial noise for non-autoencoders which use transformations such as
colorization and super-resolution. Mean squared error is based on the original output image
instead of the original input.
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

        if params["pkeep"] == "None":
            pkeep = tf.placeholder(tf.int32, (None))  # Not used
        else:
            pkeep = graph.get_tensor_by_name(params['pkeep'])

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

                    # Output variables
                    fY = np.array( sess.run([output], feed_dict={input: np.array([x_org]), pkeep: 1.0}) )

                    true_Y = tf.placeholder(tf.float32, name="true_Y", shape=fY.shape)
                    sess.run(true_Y, feed_dict={true_Y: fY})

                    loss = tf.reduce_mean(tf.squared_difference(true_Y, output))
                    grad = tf.gradients(loss, input)
                    used_pixels = []

                    ii_imag += 1
                    for iter in range(num_iterations):

                        print("{} / {} Iterations".format(iter + 1, num_iterations))

                        prod = np.prod(np.array(x).shape)
                        _x = np.reshape(x, prod)

                        fX = np.array(x).reshape(np.array([x_org]).shape)

                        # Only correct for first linearation
                        if iter == 0:
                            print("Starting Point Initialization")
                            s = np.prod(fX.shape)
                            p = np.random.uniform(-epsilon/100000, epsilon/100000, size=s)
                            fX = np.array(fX) + np.reshape(p, np.array(fX).shape)

                        dL = sess.run([grad], feed_dict={input: fX, pkeep: 1.0, true_Y: fY})
                        dL = np.reshape(dL, fX.shape)
                        _dL = np.reshape(dL, prod)
                        dL_norm = helpers.l2_norm(dL)
                        _eta = np.zeros(prod)

                        if norm == "linf":
                            _eta = (epsilon / num_iterations) * np.sign(_dL)
                        elif norm == "l2":
                            _eta = (epsilon / num_iterations) * (_dL / dL_norm)
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
                                for ww in range(W):
                                    norms[hh, ww] = np.sum(np.abs(dL[:, hh, ww, :]))
                                    if (norms[hh, ww] >= tmp_norm) and ([hh, ww] not in used_pixels):
                                        h_opt = hh
                                        w_opt = ww
                                        n_opt = epsilon * np.sign(dL[:, hh, ww, :])
                                        tmp_norm = norms[hh, ww]

                            used_pixels.append([h_opt, w_opt])
                            eta = np.zeros([H, W, D])
                            eta[h_opt, w_opt, :] = n_opt
                            _eta = eta.flatten()

                        _x = _x + _eta

                        #print(np.array(_eta).tolist())

                        x = np.reshape(_x, np.array([x_org]).shape)

                    fooledY = np.array([ sess.run([output], feed_dict={input: np.array(x), pkeep: 1.0}) ])

                    if params["colors_output"] == "cbcr":
                        X_org, X_new, fY, fooledY = helpers.merge_color_channels(params, v, x[0], fY, fooledY)
                    else:
                        # Images
                        X_org = np.array([[x_org]])
                        X_new = np.array([x])
                        fY = np.array(fY)
                        fooledY = np.array(fooledY)

                    # Mean squared error
                    mse.append(helpers.mse(fY, fooledY))

                    # Convert images to plot properly for superresolution
                    if params["description"] == "superresolution":
                        X_org, X_new, fY, fooledY = helpers.adjust_images(X_org, X_new, fY, fooledY[0], 200)

                    # Images
                    images.append([X_org, X_new, fY, fooledY])

                avg_mse = np.mean(mse)
                psnr = helpers.psnr(avg_mse)
                _psnr.append(psnr)
                _images.append(images)

        elif method == "rand":

            for epsilon in epsilon_range:

                mse = []
                images = []

                for v in value:

                    # Modify input images
                    if params["colors_input"] == "y":
                        _v = v[:,:,0]
                        _v = np.reshape(_v, [params["image_dims"][0], params["image_dims"][1], 1])
                    else:
                        _v = v

                    # Adversarial noise
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
                    elif  norm == 'pixel':
                        eta  = np.zeros(_v.shape)
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

                    _fooledY = np.array([ sess.run([output], feed_dict={input: np.array([_w]), pkeep: 1.0}) ])
                    _fY = np.array( sess.run([output], feed_dict={input: np.array([_v]), pkeep: 1.0}) )

                    # Merge color channels
                    if params["colors_output"] == "cbcr":
                        X_org, X_new, fY, fooledY = helpers.merge_color_channels(params, v, _w, _fY, _fooledY)
                    else:
                        # Images
                        X_org = np.array([[_v]])
                        X_new = np.array([[_w]])
                        fY = np.array(_fY)
                        fooledY = np.array(_fooledY)

                    # Calculate mean squared error before padding conversion
                    mse.append(helpers.mse(fY, fooledY))

                    # Convert images to plot properly for superresolution
                    if params["description"] == "superresolution":
                        X_org, X_new, fY, fooledY = helpers.adjust_images(X_org, X_new, fY, fooledY[0], 200)

                    # Images for plotting
                    images.append([X_org, X_new, fY, fooledY])

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
    if params["dataset"] == "set14":
        x_test = datasets.Dataset(params).set14(scale=2)
    elif params["dataset"] == "stl10":
        params["image_dims"] = [96, 96, 3]
        if params["colors_output"] == "rgb":
            x_test = datasets.Dataset(params).stl10(colors="rgb")
            x_test = x_test.reshape([-1, 96, 96, 3])
        elif params["colors_output"] == "cbcr":
            x_test = datasets.Dataset(params).stl10(colors="ycbr")
            x_test = x_test.reshape([-1, 96, 96, 3])

    # Import test image
    test_image = x_test[0:10]

    # Get noise
    adv_dict = {
        'psnr_input': 0, 'rand-linf': 1, 'rand-l2': 2,
        'linear-linf-1': 3, 'linear-linf-10': 4, 'linear-linf-20': 5,
        'linear-l2-1': 6, 'linear-l2-10': 7, 'linear-l2-20': 8,
        'quadratic-linf-1': 9, 'quadratic-linf-10': 10,
        'quadratic-l2-1': 11, 'quadratic-l2-10': 12,
        'linear-pixel-1': 13, 'linear-pixel-10': 14,
        'quadratic-pixel-1': 15,
        'rand-pixel-1':16, 'rand-pixel-10':17
    }
    img_mtx = {}

    if params["norm"] == "l2":
        # Epsilon
        epsilon_range = np.array(np.logspace(-0.5, 1.5, 5)) # for l2
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

        helpers.save_image_fig(img_mtx, params, adv_mtx[:, adv_dict['psnr_input']],
            ["rand-l2", 'linear-l2-1', 'linear-l2-10', 'linear-l2-20'], "l2", img_size=params["image_dims"])

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
        epsilon_range = np.array(np.logspace(-1, 2, 5)) # for l2
        adv_mtx = np.zeros([len(epsilon_range), len(adv_dict)])

        # L2
        adv_mtx[:, adv_dict['psnr_input']] = -20. * np.log10(epsilon_range)

        adv_mtx[:, adv_dict['rand-linf']], img_mtx['rand-linf'] = \
            adv_noise(test_image, params, epsilon_range, "rand", "linf", False, 1)
        adv_mtx[:, adv_dict['linear-linf-10']], img_mtx['linear-linf-10'] = \
            adv_noise(test_image, params, epsilon_range, "linear", "linf", False, 10)
        adv_mtx[:, adv_dict['linear-pixel-10']], img_mtx['linear-pixel-10'] = \
            adv_noise(test_image, params, epsilon_range, "linear", "pixel", False, 100)

        helpers.save_image_fig(img_mtx, params, adv_mtx[:, adv_dict['psnr_input']],
                                 ['rand-pixel-10', 'linear-pixel-10'], "pixel",
                                 img_size=params["image_dims"])

    np.savetxt(params['output_dir'] + '/summary/' + params['model'] + '_psnr_summary_' + params['dataset']  \
               + '_' + params["norm"] + '.csv',
               adv_mtx, delimiter=";")
    helpers.save_psnr_fig(adv_mtx, './results/images/' + params['figs_dir'] + '/' + params['model'] + '_fig_' + params[
        'dataset'] + '_' + params["norm"] + '.eps',
                          adv_dict, legend=True)

if __name__ == '__main__':
    main()
