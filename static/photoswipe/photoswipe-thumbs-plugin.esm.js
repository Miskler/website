/**
 * photoswipe-thumbs-plugin.esm.js
 * ----------------------------------------
 * PhotoSwipe v5.x plugin that adds thumbnail slider. (Tested with v5.4.4)
 *
 * Usage:
 *   import { PhotoSwipeThumbsPlugin } from './photoswipe-thumbs-plugin.esm.js';
 *   const thumbs = new PhotoSwipeThumbsPlugin(lightbox, { options });
 *
 * Author: apptaro
 * License: MIT
 * Repository: https://github.com/apptaro/photoswipe-thumbs-plugin
 */

export default class PhotoSwipeThumbsPlugin {
  /**
   * @param {PhotoSwipeLightbox} lightbox - PhotoSwipe lightbox instance
   * @param {Object} [options={}] - configuration options
   */
  constructor(lightbox, options = {}) {
    this.lightbox = lightbox;
    this.options = {
      thumbHeight: options?.thumbHeight ?? 64,
      thumbAspectRatio: options?.thumbAspectRatio ?? 1,
      gap: options?.gap ?? 6,
      paddingX: options?.paddingX ?? 0,
      dragClickThreshold: options?.dragClickThreshold ?? 6,
      autoHideOnSingle: options?.autoHideOnSingle ?? true,
      showHideAnimationDuration: options?.showHideAnimationDuration ?? 333,
      scrollAnimationDuration: options?.scrollAnimationDuration ?? 300,
      classPrefix: options?.classPrefix ?? 'pswp-thumbs',
    };

    this._thumbHeight = this.options.thumbHeight;
    this._thumbWidth = Math.round(this._thumbHeight * this.options.thumbAspectRatio);
    this._sliderHeight = this._thumbHeight + this.options.gap;
    this._scrollbarHeight = this._detectScrollbarHeight();
    this._ui = null;
    this._track = null;
    this._thumbs = [];
    this._activeThumbIdx = -1;
    this._dragMovedPx = 0;
    this._dragClickThreshold = this.options.dragClickThreshold;
    this._dragEventListeners = null;
    this._smoothScrollAnimation = null;
    this._inertiaScrollAnimation = null;
    this._origPaddingFn = null;
    this._origPaddingFnSaved = false;

    this._registerUI();
  }

  _registerUI() {
    const { lightbox, options: cfg } = this;

    const applyPaddingHook = () => {
      const pswp = lightbox.pswp;
      if (!this._origPaddingFnSaved) {
        this._origPaddingFn = pswp.options.paddingFn || null;
        this._origPaddingFnSaved = true;
      }
      pswp.options.paddingFn = (vp, item, idx) => {
        const base = this._origPaddingFn ? this._origPaddingFn(vp, item, idx) : { top:0, right:0, bottom:0, left:0 };
        return {
          top:    (base.top    || 0) + this._measureTopUIHeight(),
          right:  (base.right  || 0),
          bottom: (base.bottom || 0) + this._measureThumbsUIHeight() + (window.innerHeight - document.documentElement.clientHeight),
          left:   (base.left   || 0),
        };
      };
    };

    const unapplyPaddingHook = (pswp) => {
      if (this._origPaddingFnSaved) {
        pswp.options.paddingFn = this._origPaddingFn || undefined;
        this._origPaddingFnSaved = false;
      }
    };

    const applyCSS = () => {
      const pswp = lightbox.pswp;
      const scopeId = `pswp-th_${Math.random().toString(36).slice(2)}`;
      const scope = `[data-th="${scopeId}"]`;
      const css = `
        ${scope} .${cfg.classPrefix}-root {
          position: absolute; left: 0; right: 0; bottom: 0;
          height: calc(${this._sliderHeight}px + ${this._scrollbarHeight}px);
          padding: 0 ${cfg.paddingX}px 0;
          box-sizing: border-box;
          pointer-events: auto;
          transform: translateY(8px);
          opacity: 0;
          transition: transform ${cfg.showHideAnimationDuration}ms ease, opacity ${cfg.showHideAnimationDuration}ms ease;
          z-index: 90;
        }
        @supports (padding: max(0px)) {
          ${scope} .${cfg.classPrefix}-root { padding-bottom: max(env(safe-area-inset-bottom), 0px); }
        }
        ${scope} .${cfg.classPrefix}-root.is-visible { transform: translateY(0); opacity: 1; }
        ${scope} .${cfg.classPrefix}-track {
          height: 100%;
          overflow-x: auto; overflow-y: hidden;
          scrollbar-gutter: stable;
          display: flex; gap: ${cfg.gap}px; align-items: center;
          -webkit-overflow-scrolling: touch; scrollbar-width: thin;
          overscroll-behavior: contain;
          touch-action: pan-x;
        }
        ${scope} .${cfg.classPrefix}-track.is-dragging { user-select: none; }
        ${scope} .${cfg.classPrefix}-thumb {
          flex: 0 0 ${this._thumbWidth}px; width: ${this._thumbWidth}px; height: ${this._thumbHeight}px; border: 0; padding: 0; background: none; position: relative; border-radius: 0; outline: none; cursor: pointer;
        }
        ${scope} .${cfg.classPrefix}-img {
          position: absolute; inset: 0;
          background-size: cover; background-position: center;
          border-radius: 0; box-shadow: 0 0 0 1px rgba(255,255,255,0.4) inset;
          transition: box-shadow 160ms ease, transform 160ms ease;
          pointer-events: none;
        }
        ${scope} .${cfg.classPrefix}-thumb:is(:hover,:focus) .${cfg.classPrefix}-img {
          box-shadow: 0px 0px 0px 2px rgba(255,255,255,0.75) inset; transform: translateY(-1px);
        }
        ${scope} .${cfg.classPrefix}-thumb.is-active .${cfg.classPrefix}-img {
          box-shadow: 0px 0px 0px 2px #fff;
        }
      `;
      this._styleEl = document.createElement('style');
      this._styleEl.textContent = css;
      document.head.appendChild(this._styleEl);
      pswp.element.setAttribute('data-th', scopeId);
    };

    const unapplyCSS = (pswp) => {
      pswp.element.removeAttribute('data-th');
      if (this._styleEl) {
        document.head.removeChild(this._styleEl);
        this._styleEl = null;
      }
    };

    const buildUI = () => {
      const pswp = lightbox.pswp;
      const numItems = pswp.getNumItems();

      this._ui = document.createElement('div');
      this._ui.className = `${cfg.classPrefix}-root`;
      pswp.element.appendChild(this._ui);

      this._track = document.createElement('div');
      this._track.className = `${cfg.classPrefix}-track`;
      this._ui.appendChild(this._track);

      this._thumbs = [];
      for (let i = 0; i < numItems; i++) {
        const data = pswp.getItemData(i) || {};
        const src = data.msrc ? data.msrc : data.src;

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `${cfg.classPrefix}-thumb`;
        btn.dataset.index = String(i);
        this._track.appendChild(btn);

        const inner = document.createElement('span');
        inner.className = `${cfg.classPrefix}-img`;
        if (src) inner.style.backgroundImage = `url("${src}")`;
        btn.appendChild(inner);

        this._thumbs.push(btn);
      }

      this._track.addEventListener('click', (e) => {
        const elm = document.elementFromPoint(e.clientX, e.clientY);
        if (elm.matches(`button.${cfg.classPrefix}-thumb`)) {
          if (this._dragMovedPx > this._dragClickThreshold) {
            e.preventDefault();
            e.stopPropagation();
          } else {
            const idx = Number(elm.dataset.index) || 0;
            lightbox.pswp.goTo(idx);
          }
        }
      });

      this._activeThumbIdx = -1;
      lightbox.on('change', onChangeSlide);
      onChangeSlide();

      bindDrag();
    };

    const unbuildUI = (pswp) => {
      if (this._smoothScrollAnimation?.stop) this._smoothScrollAnimation.stop();
      if (this._inertiaScrollAnimation?.stop) this._inertiaScrollAnimation.stop();

      unbindDrag();

      lightbox.off('change', onChangeSlide);

      this._ui.remove();
      this._ui = null;
      this._track = null;
      this._thumbs = [];
    };

    const onChangeSlide = () => {
      const idx = lightbox.pswp.currIndex;
      const initial = (this._activeThumbIdx === -1)
      if (!initial) {
        if (this._activeThumbIdx === idx) return;
        this._thumbs[this._activeThumbIdx].classList.remove('is-active');
      }

      const btn = this._thumbs[idx];
      btn.classList.add('is-active');
      this._activeThumbIdx = idx;
      scrollThumbIntoView(btn, !initial);
    };

    const showUI = () => {
      requestAnimationFrame(() => {
        this._ui.classList.add('is-visible');
        pswp.updateSize(true);
      });
    };

    const hideUI = (pswp) => {
      this._ui.classList.remove('is-visible');
    };

    const scrollThumbIntoView = (btn, animate) => {
      if (this._smoothScrollAnimation?.stop) this._smoothScrollAnimation.stop();
      if (this._inertiaScrollAnimation?.stop) this._inertiaScrollAnimation.stop();
      const track = this._track;
      const rect = btn.getBoundingClientRect();
      const tRect = track.getBoundingClientRect();
      const btnMid = rect.left + rect.width / 2;
      const tMid = tRect.left + tRect.width / 2;
      const delta = btnMid - tMid;
      const target = track.scrollLeft + delta;

      if (!animate || (cfg.scrollAnimationDuration === 0)) {
        const max = track.scrollWidth - track.clientWidth;
        const dest = Math.max(0, Math.min(max, target));
        track.scrollLeft = dest;
      } else {
        smoothScrollTo(target, cfg.scrollAnimationDuration);
      }
    };

    const smoothScrollTo = (target, duration) => {
      const track = this._track;
      const startLeft = track.scrollLeft;
      const max = track.scrollWidth - track.clientWidth;
      const dest = Math.max(0, Math.min(max, target));
      const start = performance.now();

      const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

      let rafId = 0;
      const step = (now) => {
        const p = Math.min(1, (now - start) / duration);
        const v = ease(p);
        const next = startLeft + (dest - startLeft) * v;
        track.scrollLeft = Math.max(0, Math.min(max, next));
        if (p < 1) {
          rafId = requestAnimationFrame(step);
        } else {
          this._smoothScrollAnimation = null;
        }
      };
      rafId = requestAnimationFrame(step);
      this._smoothScrollAnimation = {
        stop: () => {
          cancelAnimationFrame(rafId);
          this._smoothScrollAnimation = null;
        }
      };
    };

    const bindDrag = () => {
      const track = this._track;
      const state = { active: false, startX: 0, lastX: 0, vx: 0, rafId: 0, pointerId: null };

      const onPointerDown = (e) => {
        if ((e.button !== undefined) && (e.button !== 0)) return; // only main button
        if (e.isPrimary === false) return; // only primary pointer
        if (this._smoothScrollAnimation?.stop) this._smoothScrollAnimation.stop();
        if (this._inertiaScrollAnimation?.stop) this._inertiaScrollAnimation.stop();
        state.active = true;
        state.startX = e.clientX;
        state.lastX = state.startX;
        state.vx = 0;
        track.classList.add('is-dragging');
        this._dragMovedPx = 0;
        track.setPointerCapture?.(e.pointerId);
        state.pointerId = e.pointerId;
        this._dragClickThreshold = (e.pointerType === 'touch') ? (cfg.dragClickThreshold * 1.5) : cfg.dragClickThreshold;
      };

      const onPointerMove = (e) => {
        if (!state.active) return;
        const x = e.clientX;
        const dx = x - state.lastX;
        state.lastX = x;
        state.vx = dx;
        const max = track.scrollWidth - track.clientWidth;
        const next = track.scrollLeft - dx;
        track.scrollLeft = Math.max(0, Math.min(max, next));
        this._dragMovedPx += Math.abs(dx);
        e.preventDefault();
      };

      const onPointerFinish = () => {
        if (!state.active) return;
        state.active = false;
        track.classList.remove('is-dragging');
        track.releasePointerCapture?.(state.pointerId);
        state.pointerId = null;
        const step = () => {
          state.vx *= 0.94;
          if (Math.abs(state.vx) >= 0.3) {
            const max = track.scrollWidth - track.clientWidth;
            const next = track.scrollLeft - state.vx;
            track.scrollLeft = Math.max(0, Math.min(max, next));
            state.rafId = requestAnimationFrame(step);
          } else {
            this._inertiaScrollAnimation = null;
          }
        };
        state.rafId = requestAnimationFrame(step);
        this._inertiaScrollAnimation = {
          stop: () => {
            cancelAnimationFrame(state.rafId);
            this._inertiaScrollAnimation = null;
          }
        };
      };

      track.addEventListener('pointerdown', onPointerDown);
      track.addEventListener('pointermove', onPointerMove);
      track.addEventListener('pointerup', onPointerFinish);
      track.addEventListener('pointercancel', onPointerFinish);
      this._dragEventListeners = { onPointerDown, onPointerMove, onPointerFinish };
    }

    const unbindDrag = () => {
      const track = this._track;
      const { onPointerDown, onPointerMove, onPointerFinish } = this._dragEventListeners;
      track.removeEventListener('pointerdown', onPointerDown);
      track.removeEventListener('pointermove', onPointerMove);
      track.removeEventListener('pointerup', onPointerFinish);
      track.removeEventListener('pointercancel', onPointerFinish);
      this._dragEventListeners = null;
    };

    //

    lightbox.on("uiRegister", () => { // use uiRegister because we need to apply padding hook
      const pswp = lightbox.pswp;
      const numItems = pswp.getNumItems();
      if (cfg.autoHideOnSingle && (numItems <= 1)) return;

      applyPaddingHook();
      applyCSS();
      buildUI();

      lightbox.on('openingAnimationStart', () => {
        showUI();
      });

      lightbox.on('closingAnimationStart', () => {
        hideUI(pswp);
      });

      pswp.on('destroy', () => {
        unbuildUI(pswp);
        unapplyCSS(pswp);
        unapplyPaddingHook(pswp);
      });
    });
  }

  _detectScrollbarHeight() {
    const outer = document.createElement('div');
    outer.style.cssText = 'visibility:hidden;position:absolute;left:-9999px;overflow:scroll;width:100px;height:100px;';
    const inner = document.createElement('div');
    inner.style.cssText = 'height:200px';
    outer.appendChild(inner);
    document.body.appendChild(outer);
    const height = outer.offsetHeight - outer.clientHeight;
    outer.remove();
    return Math.max(0, height || 0);
  }

  _measureTopUIHeight() {
    const pswp = this.lightbox.pswp;
    const root = pswp.element;
    const selectors = ['.pswp__top-bar', '[data-pswp-topbar]', '.pswp__caption--top'];
    let totalHeight = 0;
    for (const selector of selectors) {
      const elm = root.querySelector(selector);
      if (elm) {
        const style = getComputedStyle(elm);
        const visible = (style.display !== 'none') && (style.visibility !== 'hidden') && (parseFloat(style.opacity || '1') > 0);
        if (visible) {
          const height = Math.round(elm.getBoundingClientRect().height);
          if (height > 0) totalHeight += height;
        }
      }
    }
    return totalHeight;
  }

  _measureThumbsUIHeight() {
    if (this._ui) {
      const height = Math.round(this._ui.getBoundingClientRect().height);
      return height;
    } else {
      return this._sliderHeight + this._scrollbarHeight;
    }
  }
}
