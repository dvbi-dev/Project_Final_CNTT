/**
 * Define actions to manage tip section
 */
(function () {
  'use strict';

  function tipPanel() {
    const defaultTips = [
      'Tip: use arrows to move a selected object by 1 pixel!',
      'Tip: Shift + Click to select and modify multiple objects!',
      'Tip: hold Shift when drawing a line for 15° angle jumps!',
      'Tip: Ctrl +/-, Ctrl + wheel to zoom in and zoom out!',
    ]
    const _self = this;
    $(`${this.containerSelector} .canvas-holder .content`).append(`
    <div id="tip-container">${defaultTips[parseInt(Math.random() * defaultTips.length)]}</div>`)
    this.hideTip = function () {
      $(`${_self.containerSelector} .canvas-holder .content #tip-container`).hide();
    }

    this.showTip = function () {
      $(`${_self.containerSelector} .canvas-holder .content #tip-container`).show();
    }

    this.updateTip = function (str) {
      typeof str === 'string' && $(`${_self.containerSelector} .canvas-holder .content #tip-container`).html(str);
    }
  }

  window.ImageEditor.prototype.initializeTipSection = tipPanel;
})();