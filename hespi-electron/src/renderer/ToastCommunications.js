import React, { useContext } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

/**
 * I MIGHT have overengineered this...
 * Basically this is a wrapper where you pass a HESPIToast which has a react component and some props
 * This wrapper will take care of getting the closeToast() prop from the ToastContainer
 * So it'll pass them to the HESPIToast.
 *
 * @param {closeToast} hook passed by toastify to close the toast
 * @param {toastProps} props related to the toast passed by toastify
 * @param {toastComponent} HESPIToast this is just the instance of the HESPIToast that holds info such as the react component itself
 * as well as the properties related to the toast (e.g type, autoClose, draggable) and the react component props (such as callbacks, infos etc...)
 * @returns
 */
const ToastWrapper = (props) => {
  const { closeToast, toastProps, toastComponent } = props;
  return (
    <>
      {toastComponent.component({
        closeToast,
        toastProps,
        ...toastComponent.componentProps,
      })}
    </>
  );
};
class HESPIToast {
  constructor(component, toastProps = {}, componentProps = {}) {
    this.component = component;
    this.toastProps = toastProps;
    this.componentProps = componentProps;
  }
  show(props = {}) {
    this.componentProps = { ...this.componentProps, ...props };
    let toastFun = toast;
    if (this.toastProps?.type == 'loading') {
      toastFun = toast.loading;
    }
    toastFun(<ToastWrapper toastComponent={this} />, this.toastProps);
  }
  dismiss() {
    if (this.toastProps?.toastId) {
      toast.dismiss(this.toastProps?.toastId);
    }
  }
}

const TOASTS = {};
TOASTS.NEW_PREDICTION = new HESPIToast(
  (props) => {
    return (
      <div className="toast-msg">
        <div>
          Running new prediction
          <span
            className="test"
            style={{
              fontWeight: 'bold',
              fontSize: '1.1em',
            }}
          >
            {' '}
            {props.role}{' '}
          </span>{' '}
        </div>
      </div>
    );
  },
  {
    type: 'info',
    autoClose: 3000,
    toastId: 'newPredictionToast',
  },
);

TOASTS.PREDICTION_LOADED = new HESPIToast(
  ({file}) => {
    return <div className="toast-msg">Prediction file loaded!</div>;
  },
  {
    type: 'success',
    toastId: 'predictionLoaded',
    autoClose: 5000,
  },
);

TOASTS.PREDICTION_LOAD_FAILED = new HESPIToast(
  (props) => {
      return (
        <div className="toast-msg">
          <p>{`Failed to load prediction from file: ${props.file.name}`}</p>
        </div>
      );
  },
  {
    type: 'error',
    toastId: 'predictionLoadFailed',
    autoClose: 5000,
  },
);

TOASTS.WAITING = new HESPIToast(
  (props) => {
    return (
      <div className="toast-msg">
        <span>Waiting...:&nbsp;</span>
        <a href={window.location.href}>{window.location.href}</a>
      </div>
    );
  },
  {
    type: 'loading',
    toastId: 'waiting',
  },
);


function ToastCommunications({ closeToast, toastProps }) {
  return (
    <ToastContainer
      position="bottom-right"
      autoClose={100000}
      hideProgressBar={false}
      newestOnTop={false}
      closeOnClick={false}
      rtl={false}
      pauseOnFocusLoss={true}
      draggable={false}
      pauseOnHover={true}
      theme="light"
    />
  );
}

export { ToastCommunications, TOASTS };
