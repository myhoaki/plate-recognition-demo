(() => {
  document.querySelectorAll('[data-raw-gallery]').forEach(gallery => {
    const select = document.getElementById(gallery.dataset.select);
    const filename = gallery.querySelector('.gallery-filename');
    const position = gallery.querySelector('.gallery-position');
    const previous = gallery.querySelector('.gallery-prev');
    const next = gallery.querySelector('.gallery-next');
    let images;

    try {
      images = JSON.parse(gallery.dataset.images);
    } catch {
      images = [];
    }
    if (!select || !images.length) {
      return;
    }

    let index = images.indexOf(select.value);
    if (index < 0) index = 0;
    previous.disabled = images.length < 2;
    next.disabled = images.length < 2;

    function showImage() {
      const name = images[index];
      select.value = name;
      filename.textContent = name;
      position.textContent = `${String(index + 1).padStart(2, '0')} / ${String(images.length).padStart(2, '0')}`;
      select.dispatchEvent(new Event('change', { bubbles: true }));
    }

    previous.addEventListener('click', () => {
      index = (index - 1 + images.length) % images.length;
      showImage();
    });
    next.addEventListener('click', () => {
      index = (index + 1) % images.length;
      showImage();
    });
    select.addEventListener('change', () => {
      const selectedIndex = images.indexOf(select.value);
      if (selectedIndex >= 0 && selectedIndex !== index) {
        index = selectedIndex;
        showImage();
      }
    });
    showImage();
  });
})();
