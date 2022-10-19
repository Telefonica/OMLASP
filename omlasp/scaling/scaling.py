import os
import sys
# Delete debug TF messages
"""
    0 = all messages are logged (default behavior)
    1 = INFO messages are not printed
    2 = INFO and WARNING messages are not printed
    3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_constrained_optimization as tfco
import cv2
import numpy as np
import argparse
from pathlib import Path
from matplotlib import pyplot as plt


class Problem(tfco.ConstrainedMinimizationProblem):
  def __init__(self, S, T, method, epsilon):
    self._delta = tf.Variable(tf.expand_dims(tf.ones(shape=S.shape, dtype=tf.float32), axis=0), trainable=True, name="delta")
    self._S = tf.expand_dims(tf.convert_to_tensor(S, dtype=tf.float32) / 255, axis=0)
    self._T = tf.expand_dims(tf.convert_to_tensor(T, dtype=tf.float32) / 255, axis=0)
    self._method = method
    self._epsilon= epsilon
    self._delta_shape = self._S.shape
    self._main_constraint = 0
    self._n_constraints = self._T.shape[0] * self._T.shape[1] * self._T.shape[2] * self._T.shape[3] * 2 + 1
  
  @property
  def num_constraints(self):
    return self._n_constraints
  
  def objective(self):
    return tf.norm(self._delta, ord='euclidean')
  
  def constraints(self):
    D = tf.image.resize(self._S + self._delta, size=[self._T.shape[1], self._T.shape[2]], method=self._method)
    main_contraint = tf.norm((D - self._T), ord='euclidean') - self._epsilon
    self._main_constraint = main_contraint
    main_contraint = tf.reshape(main_contraint, shape=(1,1))
    gt_or_eq_to_zero= tf.reshape(-D, shape=(D.shape[0] * D.shape[1] * D.shape[2] * D.shape[3], 1))
    lt_or_eq_to_1= tf.reshape(D - 1, shape=(D.shape[0] * D.shape[1] * D.shape[2] * D.shape[3], 1))

    constraints = tf.concat([main_contraint, gt_or_eq_to_zero, lt_or_eq_to_1], axis=0)
    constraints = tf.reshape(constraints, shape=(constraints.shape[0]))
    return constraints


def optimize(S, T, method, epsilon, iterations):
  problem = Problem(S, T, method, epsilon)

  optimizer = tfco.LagrangianOptimizerV2(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1),
    num_constraints=problem.num_constraints
  )

  tfco.RateMinimizationProblem
  var_list = [problem._delta, optimizer.trainable_variables()]

  for i in range(iterations):
    optimizer.minimize(problem, var_list=var_list)
    if i % 1000 == 0:
      print(f'[*] Step = {i}')
      print("\t\t[*] Modification size: ", problem.objective().numpy())
      print("\t\t[*] Difference between the modified image (A) and target image: ", problem._main_constraint.numpy())


  # Modified image
  A = problem._S[0] + problem._delta[0]

  return A



def main():
  parser = argparse.ArgumentParser(
      prog='python scaling.py', 
      formatter_class=argparse.RawDescriptionHelpFormatter,
      description="\n\n\n\n" +
          "|-----------------------------------------------------------------------------|\n" +
          "|                      Tool to audit scaling methods                          |\n" +
          "|-----------------------------------------------------------------------------|" +
          "\n" +
          "\tYou can do the following tasks:\n" +
              "\t\t- Generating an image that if reduced to a certain size with a specific algorithm produces a totally different output\n" +
          "\tRequirements: \n" +
              "\t\t- You have to introduce 2 images: one called source image and another one called target image\n" +
              "\t\t- The larger the source image and the smaller the target image, the more likely the attack will be successful.\n" +
              "\t\t- The size of the target image should be the size of the model input image\n",
      epilog="We hope you were able to solve and verify the security of your algorithm"
  )

  parser.add_argument('source_img_path', type=str, help='Enter the path of the file that contains the source image')
  parser.add_argument('target_img_path', type=str, help='Enter the path of the file that contains the target image')
  parser.add_argument('-m', '--method', choices=['bilinear', 'lanczos3', 'lanczos5', 'bicubic', 'gaussian', 'nearest', 'area', 'mitchellcubic'], default='bilinear',
                        help='You must choose one of the following scaling algorithms')
  parser.add_argument('-s', '--target-size', dest='image_size', nargs=2, type=int, metavar=('height', 'width'), 
                        help='Enter the size of the target image. The target image will be pre-processed and resized to that size')
  parser.add_argument('-p', '--results-path', dest='results_path', type=str, default='./results/', help='Enter the path where you want to save the results')
  parser.add_argument('-e', '--epsilon', dest='epsilon', type=float, default=0.1, 
                        help='Enter how much you want to modify the images. If epsilon is small, the modifications of images will be small too. If epsilon is too small maybe the attack will not have success')
  parser.add_argument('-n', '--niterations', type=int, default=20000, 
                        help='Enter the number of iterations of the algorithm. The more iterations the algorithm will take, but the better the results will be. It is recommended to use a GPU')

  args = parser.parse_args()

  # Check the paths exist
  source_img_path = Path(args.source_img_path)
  target_img_path = Path(args.target_img_path)
  results_path = Path(args.results_path)

  if not source_img_path.is_file():
      print("[-] The source image path is not correct")
      sys.exit(1)

  if not target_img_path.is_file():
      print("[-] The target image path is not correct. It must be a directory")
      sys.exit(1)

  if not results_path.is_dir():
    os.makedirs(results_path)
    print(f"[*] The results directory has been created in {results_path}")


  # Read images
  S = cv2.cvtColor(cv2.imread(args.source_img_path, 1), cv2.COLOR_BGR2RGB)
  T = cv2.cvtColor(cv2.imread(args.target_img_path, 1), cv2.COLOR_BGR2RGB)

  if args.image_size:
    T = cv2.resize(T, (args.image_size[0], args.image_size[1]))
  
  A = optimize(S, T, args.method, args.epsilon, args.niterations)
  D = tf.image.resize(tf.expand_dims(A, axis=0), size=[T.shape[0], T.shape[1]])[0]

  # Save image
  cv2.imwrite(args.results_path + "/result.jpg", cv2.cvtColor(255*A.numpy(), cv2.COLOR_RGB2BGR))
  print("[+] The modified image have saved in the following path {}".format(args.results_path + "/result.jpg"))

  # Show the images
  fig=plt.figure(figsize=(8, 8))
  fig.add_subplot(2, 2, 1)
  plt.imshow(S)
  fig.add_subplot(2, 2, 2)
  plt.imshow(T)
  fig.add_subplot(2, 2, 3)
  plt.imshow(A)
  fig.add_subplot(2, 2, 4)
  plt.imshow(D)
  plt.show()


if __name__ == "__main__":
  main()

# python scaling.py marcos.jpg chema.jpg -m bilinear -p results/