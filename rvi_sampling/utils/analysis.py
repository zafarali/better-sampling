import pickle
import torch
import os
import gc
import matplotlib.pyplot as plt
from . import plotting, io

def analyze_samplers_rw(sampler_results,
                        args,
                        folder_name,
                        stochastic_process,
                        policy=None,
                        analytic=None):
    """
    Creates plots for samplers based on the random walk
    :param sampler_results: The list of sampler results
    :param folder_name: The name of the folder
    :param stochastic_process: The stochastic process we are visualizing
    :param args: arguments (ArgumentParser)
    :param analytic: the analytic solution
    :param policy: The policy if using RVI
    :return:
    """
    panel_size = plotting.determine_panel_size(len(sampler_results))

    plt.close()

    fig_dists = plt.figure(figsize=(8, 9)) # the posterior
    fig_traj = plt.figure(figsize=(9, 9)) # the trajectory distribution
    fig_traj_evol = plt.figure(figsize=(9, 9)) # the evolution of the trajectory distribution
    fig_weight_hists = plt.figure(figsize=(9, 4)) # shows the histogram of weights
    fig_traj_evol_succ = plt.figure(figsize=(9, 4)) # shows the successful trajectories

    hist_colors = zip(['r', 'g', 'b'], [1, 2, 3]) # loop over colors for the histogram
    kl_divergences = []
    kl_divergence = (100, 100)
    for i, sampler_result in enumerate(sampler_results):
        try:
            ax = fig_dists.add_subplot(panel_size + str(i + 1))
            ax = sampler_result.plot_distribution(args.rw_width, ax, alpha=0.7)
            if analytic is not None: ax = analytic.plot(stochastic_process.xT, ax, label='analytic', color='r')

            empirical_distribution = sampler_result.empirical_distribution(histbin_range=args.rw_width)


            if analytic is not None: kl_divergence = analytic.kl_divergence(empirical_distribution, stochastic_process.xT[0])
            if analytic is not None: kl_divergences.append('{},{}'.format(sampler_result.sampler_name, kl_divergence[0]))
            ax.set_title(sampler_result.summary_title() + '\nKL(true|est)={:3g}, KL(est|true)={:3g}'.format(*kl_divergence))
            print(sampler_result.summary('KL(true|est)={:3g}, KL(obs|est)={:3g}'.format(*kl_divergence)))
        except Exception as e:
            print('Could not plot posterior distribution {}'.format(e))

        ax = fig_traj.add_subplot(panel_size + str(i + 1))
        ax = sampler_result.plot_mean_trajectory(ax=ax)
        ax.set_title('Trajectory Distribution\nfor {}'.format(sampler_result.sampler_name))

        ax = fig_traj_evol.add_subplot(panel_size + str(i + 1))
        ax = sampler_result.plot_all_trajectory_evolution(ax=ax)
        ax.set_title('Evolution of Trajectories\nfor {}'.format(sampler_result.sampler_name))
        if folder_name is not None: sampler_result.save_results(folder_name)

        ax = fig_traj_evol_succ.add_subplot((panel_size + str(i + 1)))
        ax = sampler_result.plot_trajectory_evolution(ax=ax)
        ax.set_title('Successful Trajectories over time\nfor {}'.format(sampler_result.sampler_name))
        try:
            if sampler_result.sampler_name == 'RVISampler':
                fig_RL = plt.figure(figsize=(8, 4))
                ax = fig_RL.add_subplot(121)
                sampler_result.plot_reward_curves(ax)
                ax = fig_RL.add_subplot(122)
                sampler_result.plot_loss_curves(ax)
                fig_RL.tight_layout()
                if folder_name is None:
                    continue

                fig_RL.savefig(os.path.join(folder_name,'./RL_results.pdf'))
                fig_RL.clf()
                plt.close()
                gc.collect()

                torch.save({
                    'rewards_per_episode':sampler_result.rewards_per_episode,
                    'loss_per_episode':sampler_result.loss_per_episode,
                }, os.path.join(folder_name, './RL_results.pyt'))
        except AttributeError as e:
            pass
        try:
            if sampler_result._importance_sampled:
                c, j = next(hist_colors)
                ax = fig_weight_hists.add_subplot('12' + str(j))
                sampler_result.plot_posterior_weight_histogram(ax, color=c, label='{}'.format(sampler_result.sampler_name))
                ax.legend()
        except Exception as e:
            print('Could not plot weights histogram: {}'.format(e))

    # start saving things:
    try:
        fig_weight_hists.tight_layout()
        if folder_name:
            fig_weight_hists.savefig(os.path.join(folder_name, 'weight_distribution.pdf'))
            fig_weight_hists.clf()
            plt.close()
            gc.collect()
    except Exception as e:
        print('Could not create histogram of weights: {}'.format(e))

    if folder_name is None:
        return kl_divergences

    fig_dists.suptitle('MC_SAMPLES: {}, Analytic mean: {:3g}, Start {}, End {}'.format(args.samples,
                                                                                       analytic.expectation(stochastic_process.xT[0]) if analytic is not None else -100,
                                                                                       stochastic_process.x0,
                                                                                       stochastic_process.xT))
    io.put(os.path.join(folder_name, 'KL'), kl_divergences)
    fig_dists.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig_dists.savefig(os.path.join(folder_name, 'ending_distribution.pdf'))
    fig_dists.clf()
    plt.close()
    gc.collect()

    fig_traj.tight_layout()
    fig_traj.savefig(os.path.join(folder_name, 'trajectory_distribution.pdf'))
    fig_traj.clf()
    plt.close()
    gc.collect()

    fig_traj_evol.tight_layout()
    fig_traj_evol.savefig(os.path.join(folder_name, 'trajectory_evolution.pdf'))
    fig_traj_evol.clf()
    plt.close()
    gc.collect()

    fig_traj_evol_succ.tight_layout()
    fig_traj_evol_succ.savefig(os.path.join(folder_name, 'successful_trajectories.pdf'))
    fig_traj_evol_succ.clf()
    plt.close()
    gc.collect()

    # dump the arguments
    io.argparse_saver(os.path.join(folder_name, 'args'), args)

    # if we have given a policy, we should save it
    if policy is not None:
        torch.save(policy, os.path.join(folder_name, 'rvi_policy.pyt'))
        try:
            if args.plot_posterior:
                t, x, x_arrows, y_arrows_nn = plotting.visualize_proposal([policy], 50, 20, neural_network=True)
                f = plotting.multi_quiver_plot(t, x, x_arrows,
                                              [y_arrows_nn],
                                              ['Learned Neural Network Proposal'],
                                              figsize=(10, 5))
                f.savefig(os.path.join(folder_name, 'visualized_proposal.pdf'))
                f.clf()
                plt.close()
                gc.collect()
        except Exception as e:
            print('Could not plot proposal distribution {}'.format(e))
    # 

    return kl_divergences