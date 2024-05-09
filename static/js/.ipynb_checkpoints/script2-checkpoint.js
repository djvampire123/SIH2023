document.addEventListener('DOMContentLoaded', function () {
    const videoUpload = document.getElementById('videoUpload');
    const videoOutput1 = document.getElementById('videoOutput1');
    const videoOutput2 = document.getElementById('videoOutput2');
    const image = document.getElementById('image'); // Image element
    const imageContainer = document.querySelector('.image-container'); // Image container
    const graphContainer = document.getElementById('graphContainer'); // Container for both graphs
    const graph1 = document.getElementById('graph1');
    const graph2 = document.getElementById('graph2');
    const playButton = document.getElementById('playButton');

    let videosArePlaying = false;

    // Add event listener to handle video file selection
    videoUpload.addEventListener('change', function (event) {
        const file = event.target.files[0];
        const src = URL.createObjectURL(file);
        // Set the source for both videos
        videoOutput1.src = src;
        videoOutput2.src = src;

        // Sync the playback rate of videoOutput2 to be 2x
        videoOutput2.playbackRate = 1.0;

        // Show the image container
        imageContainer.style.display = 'block';
    });

    // Add event listener to handle video loaded metadata
    videoOutput1.addEventListener('loadedmetadata', function () {
        // Hide the placeholder when video is loaded
        this.style.display = 'block';
    });

    videoOutput2.addEventListener('loadedmetadata', function () {
        // Hide the placeholder when video is loaded
        this.style.display = 'block';
    });

    // Add event listener to handle video play/pause and synchronization
    videoOutput1.addEventListener('play', function () {
        if (!videosArePlaying) {
            videoOutput2.play();
            videosArePlaying = true;
        }
    });

    videoOutput1.addEventListener('pause', function () {
        if (videosArePlaying) {
            videoOutput2.pause();
            videosArePlaying = false;
        }
    });

    videoOutput2.addEventListener('play', function () {
        if (!videosArePlaying) {
            videoOutput1.play();
            videosArePlaying = true;
        }
    });

    videoOutput2.addEventListener('pause', function () {
        if (videosArePlaying) {
            videoOutput1.pause();
            videosArePlaying = false;
        }
    });

    playButton.addEventListener('click', function () {
        if (!videosArePlaying) {
            videoOutput1.play();
            videoOutput2.play();
            videosArePlaying = true;
        }
    });

    videoOutput1.addEventListener('click', function () {
        graph1.style.display = graph1.style.display === 'none' ? 'block' : 'none';
        displayGraph(graph1, 'Graph 1 Data');
    });

    videoOutput2.addEventListener('click', function () {
        graph2.style.display = graph2.style.display === 'none' ? 'block' : 'none';
        displayGraph(graph2, 'Graph 2 Data');
    });

    function displayGraph(graphElement, dataTitle) {
        if (graphElement.style.display === 'block') {
            const data = [{
                x: [1, 2, 3, 4, 5],
                y: [2, 3, 4, 5, 6],
                z: [3, 5, 7, 9, 11],
                type: 'scatter3d',
                mode: 'markers',
                marker: {
                    size: 12,
                    line: {
                        color: 'rgba(217, 217, 217, 0.14)',
                        width: 0.5
                    },
                    opacity: 0.8
                }
            }];

            const layout = {
                width: 500,
                height: 500,
                title: dataTitle,
                scene: {
                    xaxis: { title: 'X Axis' },
                    yaxis: { title: 'Y Axis' },
                    zaxis: { title: 'Z Axis' }
                }
            };

            Plotly.newPlot(graphElement, data, layout);

            // Center the graph
            graphContainer.style.display = 'block';

            // Hide the image container
            imageContainer.style.display = 'none';
        }
    }
});
